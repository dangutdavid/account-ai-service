require('dotenv').config();
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk'); // ✅ ADD THIS

const app = express();
const PORT = process.env.PORT || 3000;
const HOST = '0.0.0.0';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const anthropic = new Anthropic({ // ✅ ADD THIS
  apiKey: process.env.ANTHROPIC_API_KEY
});

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.get('/health', (_req, res) => {
  res.status(200).json({
    status: 'ok',
    message: 'Account AI service is running'
  });
});

// ======================================================
// MODEL NORMALIZATION
// ======================================================
function normalizeModelName(model) {
  const value = String(model || '').trim().toLowerCase();

  if (!value) return 'gpt-4.1-mini';

  // OpenAI
  if (value === 'gpt-4.1 mini' || value === 'gpt-4.1-mini') return 'gpt-4.1-mini';
  if (value === 'gpt-4.1') return 'gpt-4.1';
  if (value === 'gpt-4o mini' || value === 'gpt-4o-mini') return 'gpt-4o-mini';
  if (value === 'gpt-4o') return 'gpt-4o';

  // ✅ ADD Claude support
  if (value === 'claude-sonnet-4') return 'claude-sonnet-4';
  if (value.startsWith('claude')) return value;

  return 'gpt-4.1-mini';
}

// ======================================================
// CLAUDE MODEL MAPPING
// ======================================================
function mapClaudeModel(model) {
  const value = String(model || '').toLowerCase();

  if (value === 'claude-sonnet-4') {
    return 'claude-3-7-sonnet-latest';
  }

  return value;
}

// ======================================================
// PROMPT BUILDERS
// ======================================================
function buildSystemPrompt(persona) {
  return `
You are a Salesforce AI Account Copilot.

Persona: ${persona || 'balanced'}

Rules:
- Answer only from provided Salesforce context.
- Do not invent facts.
- If data is missing, say so clearly.
- Prefer structured business-friendly answers.
- When useful, summarize by sections:
  Account Overview
  Contacts
  Opportunities
  Cases
  Contracts
  Files
  Risks
  Recommended Next Actions
- Keep the answer concise but useful.
- Generate suggested follow-up questions based on the context.
`;
}

function buildUserPrompt(userMessage, context) {
  return `
USER QUESTION:
${userMessage}

SALESFORCE CONTEXT:
${JSON.stringify(context, null, 2)}
`;
}

// ======================================================
// CLAUDE CHAT CALL
// ======================================================
async function callClaudeChat({ model, persona, userMessage, context }) {
  const claudeModel = mapClaudeModel(model);

  const response = await anthropic.messages.create({
    model: claudeModel,
    max_tokens: 500,
    messages: [
      {
        role: 'user',
        content: `
${buildSystemPrompt(persona)}

${buildUserPrompt(userMessage, context)}
`
      }
    ]
  });

  const text = response.content?.[0]?.text || 'No response generated.';

  return {
    text,
    tokenUsage:
      (response.usage?.input_tokens || 0) +
      (response.usage?.output_tokens || 0)
  };
}

// ======================================================
// CITATIONS
// ======================================================
function buildCitations(context) {
  const citations = [];

  if (context?.contacts?.length) {
    context.contacts.slice(0, 5).forEach(c => {
      citations.push({
        objectApiName: 'Contact',
        label: c.name,
        excerpt: `${c.title || ''} ${c.email || ''}`.trim(),
        url: c.id ? `/${c.id}` : null,
        recordId: c.id || null
      });
    });
  }

  if (context?.opportunities?.length) {
    context.opportunities.slice(0, 5).forEach(o => {
      citations.push({
        objectApiName: 'Opportunity',
        label: o.name,
        excerpt: `${o.stageName || ''} ${o.amount || ''}`.trim(),
        url: o.id ? `/${o.id}` : null,
        recordId: o.id || null
      });
    });
  }

  if (context?.cases?.length) {
    context.cases.slice(0, 5).forEach(c => {
      citations.push({
        objectApiName: 'Case',
        label: c.caseNumber || c.subject,
        excerpt: `${c.subject || ''} ${c.status || ''}`.trim(),
        url: c.id ? `/${c.id}` : null,
        recordId: c.id || null
      });
    });
  }

  if (context?.contracts?.length) {
    context.contracts.slice(0, 5).forEach(c => {
      citations.push({
        objectApiName: 'Contract',
        label: c.contractNumber || 'Contract',
        excerpt: `${c.status || ''}`.trim(),
        url: c.id ? `/${c.id}` : null,
        recordId: c.id || null
      });
    });
  }

  if (context?.files?.length) {
    context.files.slice(0, 5).forEach(f => {
      citations.push({
        objectApiName: 'ContentDocument',
        label: f.title,
        excerpt: `${f.fileType || 'File'}`.trim(),
        url: f.downloadUrl || null,
        recordId: f.documentId || null
      });
    });
  }

  return citations;
}

// ======================================================
// CHAT ENDPOINT
// ======================================================
app.post('/api/v1/chat', async (req, res) => {
  try {
    const body = req.body || {};
    const userMessage = body.userMessage || 'No message provided';
    const model = normalizeModelName(body.model);
    const persona = body.persona || 'balanced';
    const context = body.context || {};

    console.log('CHAT REQUEST MODEL:', body.model, '=>', model);

    let aiResult;

    // ✅ ROUTING
    if (model.startsWith('claude')) {
      aiResult = await callClaudeChat({
        model,
        persona,
        userMessage,
        context
      });
    } else {
      const response = await client.responses.create({
        model,
        max_output_tokens: 500,
        input: [
          {
            role: 'system',
            content: buildSystemPrompt(persona)
          },
          {
            role: 'user',
            content: buildUserPrompt(userMessage, context)
          }
        ]
      });

      aiResult = {
        text: response.output_text || 'No response generated.',
        tokenUsage: response.usage?.total_tokens || 0
      };
    }

    const citations = buildCitations(context);

    return res.status(200).json({
      response: aiResult.text,
      model,
      cost: 0,
      tokenUsage: aiResult.tokenUsage,
      confidence: 90,
      suggestedQuestions: [
        'Which opportunities are at risk?',
        'What are the next best actions for the pipeline?',
        'Who are the key contacts on this account?',
        'Are there any contract renewal risks?'
      ],
      citations
    });
  } catch (error) {
    console.error('AI Chat Error:', error);

    return res.status(500).json({
      error: 'AI service failed',
      message: error.message
    });
  }
});

app.listen(PORT, HOST, () => {
  console.log(`Account AI service running on http://${HOST}:${PORT}`);
});