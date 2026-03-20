require('dotenv').config();
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');

const app = express();
const PORT = process.env.PORT || 3000;
const HOST = '0.0.0.0';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const anthropic = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  : null;

app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.get('/health', (_req, res) => {
  res.status(200).json({
    status: 'ok',
    message: 'Account AI service is running',
    openaiConfigured: Boolean(process.env.OPENAI_API_KEY),
    anthropicConfigured: Boolean(process.env.ANTHROPIC_API_KEY)
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

  // Claude
  if (value === 'claude-sonnet-4') return 'claude-sonnet-4';
  if (value.startsWith('claude')) return value;

  return 'gpt-4.1-mini';
}

// ======================================================
// CLAUDE MODEL MAPPING
// ======================================================
function mapClaudeModel(model) {
  const value = String(model || '').trim().toLowerCase();

  if (value === 'claude-sonnet-4') {
    return 'claude-3-7-sonnet-latest';
  }

  return value;
}

// ======================================================
// TIMEOUT WRAPPER
// ======================================================
async function withTimeout(promise, ms, label) {
  let timeoutId;

  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutId);
  }
}

// ======================================================
// FRIENDLY ERROR HELPERS
// ======================================================
function stringifyError(error) {
  if (!error) return 'Unknown error';

  if (typeof error === 'string') return error;

  if (error.message && typeof error.message === 'string') {
    return error.message;
  }

  try {
    return JSON.stringify(error);
  } catch (_e) {
    return 'Unknown error';
  }
}

function getProviderFromModel(model) {
  return String(model || '').startsWith('claude') ? 'Anthropic' : 'OpenAI';
}

function isAnthropicLowCreditError(error) {
  const text = stringifyError(error).toLowerCase();
  return (
    text.includes('credit balance is too low') ||
    text.includes('please go to plans & billing') ||
    text.includes('anthropic api')
  );
}

function isOpenAILowCreditError(error) {
  const text = stringifyError(error).toLowerCase();
  return (
    text.includes('insufficient_quota') ||
    text.includes('quota') ||
    text.includes('billing') ||
    text.includes('exceeded your current quota') ||
    text.includes('check your plan and billing') ||
    text.includes('rate limit reached for requests') && text.includes('billing')
  );
}

function getFriendlyProviderError(error, model) {
  const provider = getProviderFromModel(model);
  const rawMessage = stringifyError(error);

  if (provider === 'Anthropic' && isAnthropicLowCreditError(error)) {
    return {
      provider,
      code: 'LOW_CREDIT',
      message:
        'Anthropic credit is too low for Claude requests. Please top up Anthropic billing or switch to an OpenAI model.'
    };
  }

  if (provider === 'OpenAI' && isOpenAILowCreditError(error)) {
    return {
      provider,
      code: 'LOW_CREDIT',
      message:
        'OpenAI credit or quota is too low for this request. Please check OpenAI billing/usage or switch to another configured provider.'
    };
  }

  return {
    provider,
    code: 'PROVIDER_ERROR',
    message: rawMessage
  };
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

function buildRecordInsightPrompt(objectApiName, recordId, fields) {
  return `
You are a Salesforce AI Record Intelligence assistant.

Your task is to analyze a single Salesforce record and return business insight.

Object API Name: ${objectApiName}
Record Id: ${recordId}

Record Fields:
${JSON.stringify(fields, null, 2)}

Return ONLY valid JSON with this exact shape:
{
  "summary": "short business summary",
  "riskLevel": "Low | Medium | High | Critical",
  "recommendedAction": "clear next best action",
  "shouldCreateTask": true,
  "taskSubject": "short task subject",
  "taskDescription": "task details",
  "confidenceScore": 0.85
}

Rules:
- Do not include markdown fences.
- Do not include commentary outside JSON.
- Base the answer only on supplied fields.
- If data is limited, say so in the summary.
- riskLevel must be exactly one of: Low, Medium, High, Critical.
- confidenceScore must be a number between 0 and 1.
`;
}

// ======================================================
// HELPERS
// ======================================================
function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (_e) {
    return null;
  }
}

function normalizeRiskLevel(value) {
  const text = String(value || '').trim().toLowerCase();
  if (text === 'low') return 'Low';
  if (text === 'medium' || text === 'moderate') return 'Medium';
  if (text === 'high') return 'High';
  if (text === 'critical') return 'Critical';
  return 'Medium';
}

function normalizeConfidenceScore(value) {
  const num = Number(value);
  if (Number.isNaN(num)) return 0.7;
  if (num < 0) return 0;
  if (num > 1) return 1;
  return num;
}

function normalizeInsightPayload(parsed, objectApiName) {
  return {
    summary:
      parsed?.summary ||
      `AI analysis completed for ${objectApiName}, but limited structured detail was returned.`,
    riskLevel: normalizeRiskLevel(parsed?.riskLevel),
    recommendedAction:
      parsed?.recommendedAction ||
      'Review the record and determine the next best action.',
    shouldCreateTask: Boolean(parsed?.shouldCreateTask),
    taskSubject:
      parsed?.taskSubject || `Review ${objectApiName} AI insight`,
    taskDescription:
      parsed?.taskDescription ||
      'AI flagged this record for review. Please inspect the summary and decide the next step.',
    confidenceScore: normalizeConfidenceScore(parsed?.confidenceScore)
  };
}

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
// CLAUDE CHAT CALL
// ======================================================
async function callClaudeChat({ model, persona, userMessage, context }) {
  if (!anthropic) {
    throw new Error('Claude support is not configured. Missing ANTHROPIC_API_KEY.');
  }

  const claudeModel = mapClaudeModel(model);

  try {
    const response = await withTimeout(
      anthropic.messages.create({
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
      }),
      20000,
      'Claude chat request'
    );

    const text = response.content?.[0]?.text || 'No response generated.';

    return {
      text,
      tokenUsage:
        (response.usage?.input_tokens || 0) +
        (response.usage?.output_tokens || 0)
    };
  } catch (error) {
    const friendly = getFriendlyProviderError(error, model);
    const wrapped = new Error(friendly.message);
    wrapped.provider = friendly.provider;
    wrapped.code = friendly.code;
    wrapped.originalMessage = stringifyError(error);
    throw wrapped;
  }
}

// ======================================================
// RECORD INSIGHT ANALYSIS
// ======================================================
async function runRecordInsightAnalysis(body) {
  const objectApiName = body.objectApiName || 'UnknownObject';
  const recordId = body.recordId || 'UnknownRecord';
  const fields = body.fields || {};

  const prompt = buildRecordInsightPrompt(objectApiName, recordId, fields);

  try {
    const response = await withTimeout(
      client.responses.create({
        model: 'gpt-4.1-mini',
        max_output_tokens: 300,
        input: [
          {
            role: 'system',
            content: 'You analyze Salesforce records and return strict JSON only.'
          },
          {
            role: 'user',
            content: prompt
          }
        ]
      }),
      15000,
      'Analyze request'
    );

    const aiText = response.output_text || '';
    const parsed = safeJsonParse(aiText);

    if (!parsed) {
      return {
        summary: `AI analysis for ${objectApiName} completed, but the model did not return valid JSON. Raw output has been captured.`,
        riskLevel: 'Medium',
        recommendedAction: 'Review the AI output and retry if needed.',
        shouldCreateTask: false,
        taskSubject: `Review ${objectApiName} AI insight`,
        taskDescription: aiText || 'No structured AI output returned.',
        confidenceScore: 0.5
      };
    }

    return normalizeInsightPayload(parsed, objectApiName);
  } catch (error) {
    const friendly = getFriendlyProviderError(error, 'gpt-4.1-mini');
    const wrapped = new Error(friendly.message);
    wrapped.provider = friendly.provider;
    wrapped.code = friendly.code;
    wrapped.originalMessage = stringifyError(error);
    throw wrapped;
  }
}

// ======================================================
// ANALYZE ENDPOINTS
// ======================================================
app.post('/analyze', async (req, res) => {
  try {
    const result = await runRecordInsightAnalysis(req.body || {});
    return res.status(200).json(result);
  } catch (error) {
    console.error('AI Analyze Error:', error);

    return res.status(500).json({
      error: 'AI analysis failed',
      provider: error.provider || 'OpenAI',
      code: error.code || 'PROVIDER_ERROR',
      message: error.message,
      details: error.originalMessage || error.message
    });
  }
});

app.post('/api/v1/analyze', async (req, res) => {
  try {
    const result = await runRecordInsightAnalysis(req.body || {});
    return res.status(200).json(result);
  } catch (error) {
    console.error('AI Analyze Error:', error);

    return res.status(500).json({
      error: 'AI analysis failed',
      provider: error.provider || 'OpenAI',
      code: error.code || 'PROVIDER_ERROR',
      message: error.message,
      details: error.originalMessage || error.message
    });
  }
});

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

    if (model.startsWith('claude')) {
      aiResult = await callClaudeChat({
        model,
        persona,
        userMessage,
        context
      });
    } else {
      try {
        const response = await withTimeout(
          client.responses.create({
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
          }),
          20000,
          'OpenAI chat request'
        );

        aiResult = {
          text: response.output_text || 'No response generated.',
          tokenUsage: response.usage?.total_tokens || 0
        };
      } catch (error) {
        const friendly = getFriendlyProviderError(error, model);
        const wrapped = new Error(friendly.message);
        wrapped.provider = friendly.provider;
        wrapped.code = friendly.code;
        wrapped.originalMessage = stringifyError(error);
        throw wrapped;
      }
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
      provider: error.provider || getProviderFromModel(normalizeModelName(req.body?.model)),
      code: error.code || 'PROVIDER_ERROR',
      message: error.message,
      details: error.originalMessage || error.message
    });
  }
});

app.listen(PORT, HOST, () => {
  console.log(`Account AI service running on http://${HOST}:${PORT}`);
});