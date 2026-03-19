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

app.get('/health', (_req, res) => {
  res.status(200).json({
    status: 'ok',
    message: 'Account AI service is running'
  });
});

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

app.post('/analyze', async (req, res) => {
  try {
    const body = req.body || {};
    const objectApiName = body.objectApiName || 'UnknownObject';
    const recordId = body.recordId || 'UnknownRecord';
    const fields = body.fields || {};

    const prompt = buildRecordInsightPrompt(objectApiName, recordId, fields);

    const response = await client.responses.create({
      model: 'gpt-4.1-mini',
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
    });

    const aiText = response.output_text || '';
    const parsed = safeJsonParse(aiText);

    if (!parsed) {
      return res.status(200).json({
        summary: `AI analysis for ${objectApiName} completed, but the model did not return valid JSON. Raw output has been captured.`,
        riskLevel: 'Medium',
        recommendedAction: 'Review the AI output and retry if needed.',
        shouldCreateTask: false,
        taskSubject: `Review ${objectApiName} AI insight`,
        taskDescription: aiText || 'No structured AI output returned.',
        confidenceScore: 0.5
      });
    }

    return res.status(200).json(
      normalizeInsightPayload(parsed, objectApiName)
    );
  } catch (error) {
    console.error('AI Analyze Error:', error);

    return res.status(500).json({
      error: 'AI analysis failed',
      message: error.message
    });
  }
});

app.post('/api/v1/chat', async (req, res) => {
  try {
    const body = req.body || {};
    const userMessage = body.userMessage || 'No message provided';
    const model = body.model || 'gpt-4.1-mini';
    const persona = body.persona || 'balanced';
    const context = body.context || {};

    const response = await client.responses.create({
      model,
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

    const aiText = response.output_text || 'No response generated.';
    const citations = buildCitations(context);

    return res.status(200).json({
      response: aiText,
      model: model,
      cost: 0,
      tokenUsage: response.usage?.total_tokens || 0,
      confidence: 90,
      suggestedQuestions: [
        'Which opportunities are at risk?',
        'What are the next best actions for the pipeline?',
        'Who are the key contacts on this account?',
        'Are there any contract renewal risks?'
      ],
      citations: citations
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