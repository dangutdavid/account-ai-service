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
// CHAT ENDPOINT (REAL AI RESPONSE)
// ============================================================
app.post('/api/v1/chat', async (req, res) => {
  try {
    const body = req.body || {};

    const userMessage = body.userMessage || 'No message provided';
    const model = body.model || 'gpt-4.1';
    const persona = body.persona || 'balanced';
    const context = body.context || {};

    const accountName = context?.account?.name || 'this account';

    const prompt = `
You are a Salesforce Account AI Assistant.

Persona: ${persona}

Account Name: ${accountName}

Account Context:
${JSON.stringify(context, null, 2)}

User Question:
${userMessage}

Instructions:
- Answer using ONLY the context provided
- Be concise and useful
- If information is missing, say so clearly
- Do not invent facts
`;

    const response = await client.responses.create({
      model: model,
      input: prompt
    });

    const aiText = response.output_text || "No response generated.";

    return res.status(200).json({
      response: aiText,
      model: model,
      cost: 0,
      tokenUsage: response.usage?.total_tokens || 0,
      confidence: 90,
      suggestedQuestions: [
        "What open opportunities exist?",
        "Are there any active cases?",
        "Who are the key contacts?",
        "What risks exist on this account?"
      ],
      citations: []
    });

  } catch (error) {

    console.error("AI Chat Error:", error);

    return res.status(500).json({
      error: "AI service failed",
      message: error.message
    });
  }
});

// ============================================================
// START SERVER
// ============================================================
app.listen(PORT, HOST, () => {
  console.log(`Account AI service running on http://${HOST}:${PORT}`);
});