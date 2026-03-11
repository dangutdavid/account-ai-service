require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

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
// CHAT ENDPOINT (DUMMY RESPONSE FOR NOW)
// ============================================================
app.post('/api/v1/chat', (req, res) => {
  try {
    const body = req.body || {};
    const userMessage = body.userMessage || 'No message provided';
    const model = body.model || 'test-model';
    const persona = body.persona || 'balanced';
    const context = body.context || {};
    const accountName = context?.account?.name || 'this account';

    const responseText =
      `Test response for ${accountName}. ` +
      `You asked: "${userMessage}". ` +
      `Persona used: ${persona}. ` +
      `This is currently a mock response from the backend service.`;

    return res.status(200).json({
      response: responseText,
      model: model,
      cost: 0,
      tokenUsage: 0,
      confidence: 100,
      suggestedQuestions: [
        'What are the open opportunities?',
        'What support issues are active?',
        'Who are the main contacts on this account?'
      ],
      citations: []
    });
  } catch (error) {
    console.error('Chat endpoint error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

// ============================================================
// START SERVER
// ============================================================
app.listen(PORT, () => {
  console.log(`Account AI service running on http://localhost:${PORT}`);
});