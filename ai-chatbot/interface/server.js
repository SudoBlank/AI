const express = require('express');
const path = require('path');
const app = express();

// Settings
const PORT = process.env.PORT || 3000;
const STATIC_DIR = path.join(__dirname, 'web_ui');

// Middleware
app.use(express.json());
app.use(express.static(STATIC_DIR));

// Routes
app.post('/chat', async (req, res) => {
    try {
        const { message, userId } = req.body;
        
        // Get response from Python backend
        // TODO: Implement actual chatbot integration
        const response = {
            text: "This is a placeholder response",
            confidence: 0.95
        };
        
        res.json(response);
    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});