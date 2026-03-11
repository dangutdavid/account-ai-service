require('dotenv').config();
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');

const app = express();
const PORT = process.env.PORT || 3000;
const HOST = '0.0.0.0';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

app.use(cors());
app.use(express.json({ limit: '2mb' }));

// ============================================================
// HEALTH CHECK
// ============================================================
app.get('/health', (_req, res) => {
  res.status(200).json({
    status: 'ok',
    message: 'Account AI service is running'
  });
});

// ============================================================
// HELPERS
// ============================================================
function buildSystemPrompt(persona) {
  return `
You are a Salesforce CRM assistant embedded inside an Account record page.

Your job is to help users understand an Account and its related CRM records using ONLY the structured Salesforce data provided.

Persona mode: ${persona}

Rules:
- Use only the CRM context supplied in the request
- Do not invent facts
- If something is missing from the context, say that clearly
- Be practical, concise, and business-friendly
- When summarising an account, mention important pipeline, support, contact, and contract signals if available
- Highlight risks, gaps, and possible next actions
- If opportunities exist, summarise their stage, amount, probability, and close date where available
- If cases exist, mention active support issues
- If contacts exist, mention key contacts if relevant
- If contracts exist, mention contract status or dates if relevant
- Prefer bullet points when useful
`;
}

function buildUserPrompt(userMessage, context) {
  return `
User question:
${userMessage}

Structured Salesforce CRM context:
${JSON.stringify(context, null, 2)}

Please answer the user using this context.
`;
}

function buildSuggestedQuestions(context) {
  const suggestions = [];

  if (context?.opportunities?.length) {
    suggestions.push('Which opportunities are at risk?');
    suggestions.push('What are the next best actions for the pipeline?');
  }

  if (context?.cases?.length) {
    suggestions.push('Are there any active support issues?');
  }

  if (context?.contacts?.length) {
    suggestions.push('Who are the key contacts on this account?');
  }

  if (context?.contracts?.length) {
    suggestions.push('Are there any contract renewal risks?');
  }

  if (suggestions.length < 4) {
    suggestions.push('Summarise this account');
    suggestions.push('What risks exist on this account?');
    suggestions.push('What follow-up actions do you recommend?');
    suggestions.push('What should I know before meeting this account?');
  }

  return [...new Set(suggestions)].slice(0, 4);
}

// ============================================================
// CHAT ENDPOINT (REAL AI RESPONSE)
// ============================================================
app.post('/api/v1/chat', async (req, res) => {
  try {
    const body = req.body || {};

    const userMessage = body.userMessage || 'No message provided';
    const model = body.model || 'gpt-4.1-mini';
    const persona = body.persona || 'balanced';
    const context = body.context || {};

    const systemPrompt = buildSystemPrompt(persona);
    const userPrompt = buildUserPrompt(userMessage, context);

    const response = await client.responses.create({
      model,
      input: [
        {
          role: 'system',
          content: [
            {
              type: 'input_text',
              text: systemPrompt
            }
          ]
        },
        {
          role: 'user',
          content: [
            {
              type: 'input_text',
              text: userPrompt
            }
          ]
        }
      ]
    });

    const aiText = response.output_text || 'No response generated.';

    return res.status(200).json({
      response: aiText,
      model,
      cost: 0,
      tokenUsage: response.usage?.total_tokens || 0,
      confidence: 90,
      suggestedQuestions: buildSuggestedQuestions(context),
      citations: []
    });
  } catch (error) {
    console.error('AI Chat Error:', error);

    return res.status(500).json({
      error: 'AI service failed',
      message: error?.message || 'Unknown error'
    });
  }
});

// ============================================================
// START SERVER
// ============================================================
app.listen(PORT, HOST, () => {
  console.log(`Account AI service running on http://${HOST}:${PORT}`);
});