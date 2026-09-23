// server.js - Fully Resilient OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const http = require('http');
const https = require('https');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;

const MAX_RETRIES = 3;            
const INITIAL_RETRY_DELAY = 1500; 

const axiosInstance = axios.create({
  httpAgent: new http.Agent({ keepAlive: true }),
  httpsAgent: new https.Agent({ keepAlive: true }),
});

const MODEL_MAPPING = {
  'glmflash': "z-ai/glm-5.3-flash",
  'glm': "z-ai/glm-5.3",
  'nvidia': "nvidia/nemotron-3-ultra-550b-a55b"
};

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function parseAxiosStreamError(error) {
  if (error.response?.data && typeof error.response.data.on === 'function') {
    try {
      const chunks = [];
      for await (const chunk of error.response.data) {
        chunks.push(chunk);
      }
      const raw = Buffer.concat(chunks).toString('utf-8');
      try {
        const parsed = JSON.parse(raw);
        return parsed.detail || parsed.error?.message || raw;
      } catch (e) {
        return raw || error.message;
      }
    } catch (e) {
      return error.message;
    }
  }
  return error.response?.data?.detail || error.response?.data?.error?.message || error.message;
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'Resilient NIM Proxy Active' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(m => ({ id: m, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;
    
    // 🚀 Robust API Key Resolution: Railway Env -> JanitorAI Auth Header
    const incomingHeader = req.headers.authorization ? req.headers.authorization.replace(/^Bearer\s+/i, '').trim() : '';
    const rawKey = process.env.NIM_API_KEY || process.env.NVIDIA_API_KEY || incomingHeader;
    const activeKey = rawKey ? rawKey.replace(/^["']|["']$/g, '').trim() : '';

    if (!activeKey) {
      throw new Error("No API key found. Set NIM_API_KEY in Railway or enter your nvapi key in JanitorAI.");
    }

    // Clean previous thought traces
    let cleanedMessages = (messages || []).map(msg => {
      let content = msg.content;
      if (msg.role === 'assistant' && typeof content === 'string') {
        content = content.replace(/<think>[\s\S]*?<\/think>\s*/g, '');
      }
      return {
        role: msg.role,
        content: content || "" 
      };
    });

    let nimModel = MODEL_MAPPING[model] || model || 'meta/llama-3.1-8b-instruct'; 
    const safeMaxTokens = Math.min(max_tokens || 4096, 4096);

    const nimRequest = {
      model: nimModel,
      messages: cleanedMessages, 
      temperature: temperature !== undefined ? temperature : 0.6,
      max_tokens: safeMaxTokens,
      stream: stream || false
    };
    
    if (ENABLE_THINKING_MODE) {
      if (nimModel.includes('glm')) {
        nimRequest.chat_template_kwargs = { enable_thinking: true, clear_thinking: false };
      } else if (nimModel.includes('deepseek') || nimModel.includes('qwen')) {
        nimRequest.chat_template_kwargs = { thinking: true }; 
      }
    }
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      if (res.flushHeaders) res.flushHeaders();

      const heartbeat = setInterval(() => {
        try {
          res.write(': keepalive\n\n');
        } catch (err) {
          clearInterval(heartbeat);
        }
      }, 15000);

      let response;
      let currentDelay = INITIAL_RETRY_DELAY;
      
      try {
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
          try {
            response = await axiosInstance.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
              headers: { 
                'Authorization': `Bearer ${activeKey}`, 
                'Content-Type': 'application/json' 
              },
              responseType: 'stream',
              timeout: 600000
            });
            break;
          } catch (error) {
            const status = error.response?.status;
            const detailedErrorMsg = await parseAxiosStreamError(error);

            console.error(`🚨 NVIDIA NIM Error [Status ${status || 'Unknown'}]:`, detailedErrorMsg);

            // Fail immediately on client errors (401, 400, 403, 404)
            if (status === 400 || status === 401 || status === 403 || status === 404 || status === 410) {
              throw new Error(`[HTTP ${status}] ${detailedErrorMsg}`);
            }

            const is429 = status === 429 || detailedErrorMsg.includes('429') || detailedErrorMsg.includes('Rate limit');
            const isRetryable = is429 || status >= 500 || detailedErrorMsg.includes('ResourceExhausted') || detailedErrorMsg.includes('limit reached');

            if (isRetryable && attempt < MAX_RETRIES) {
              const retryAfterHeader = error.response?.headers?.['retry-after'];
              let waitTime = is429 ? Math.max(currentDelay, 5000) : currentDelay;
              if (retryAfterHeader) waitTime = (parseInt(retryAfterHeader, 10) * 1000) + 1000;

              console.warn(`⚠️ [Attempt ${attempt}/${MAX_RETRIES}] Retrying in ${waitTime / 1000}s...`);
              await sleep(waitTime);
              currentDelay *= 2;
            } else {
              throw new Error(detailedErrorMsg);
            }
          }
        }

        clearInterval(heartbeat);
        
        let buffer = '';
        let reasoningStarted = false;
        
        response.data.on('data', (chunk) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          
          lines.forEach(line => {
            if (line.startsWith('data: ') && !line.includes('[DONE]')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.choices?.[0]?.delta) {
                  const delta = data.choices[0].delta;
                  const reasoning = delta.reasoning_content;
                  const content = delta.content;
                  const isGLM = nimModel.includes('glm');

                  if (isGLM) {
                    delta.content = (content !== undefined && content !== null) ? content : "";
                    delete delta.reasoning_content;
                  } else {
                    if (SHOW_REASONING) {
                      let combinedContent = '';
                      
                      if (reasoning) {
                        if (!reasoningStarted) {
                          combinedContent += '<think>\n';
                          reasoningStarted = true;
                        }
                        combinedContent += reasoning;
                      }
                      
                      if (content !== undefined && content !== null) {
                        if (reasoningStarted && content !== '') {
                          combinedContent += '\n</think>\n\n';
                          reasoningStarted = false;
                        }
                        combinedContent += content;
                      }
                      
                      if (combinedContent !== '' || typeof content === 'string') {
                        delta.content = combinedContent || content || "";
                        delete delta.reasoning_content;
                      }
                    }
                  }
                }
                res.write(`data: ${JSON.stringify(data)}\n\n`);
              } catch (e) {
                // Incomplete JSON chunk
              }
            } else if (line.includes('[DONE]')) {
              res.write('data: [DONE]\n\n');
            }
          });
        });
        
        response.data.on('end', () => res.end());

      } catch (streamError) {
        clearInterval(heartbeat);
        console.error('🚨 Request Execution Failed:', streamError.message);
        
        // Output inside delta content so JanitorAI displays the real message instead of pgshag2
        res.write(`data: ${JSON.stringify({
          choices: [{ delta: { content: `\n\n**[NVIDIA Proxy Error]**: ${streamError.message}` } }]
        })}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }

    } else {
      const response = await axiosInstance.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: { 
          'Authorization': `Bearer ${activeKey}`, 
          'Content-Type': 'application/json' 
        },
        responseType: 'json',
        timeout: 600000 
      });
      res.json(response.data); 
    }
    
  } catch (error) {
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.response?.data?.detail || error.message || 'Server error' }
      });
    }
  }
});

app.all('*', (req, res) => res.status(404).json({ error: { message: 'Not found' } }));
app.listen(PORT, () => console.log(`🚀 Resilient Proxy running on port ${PORT}`));
