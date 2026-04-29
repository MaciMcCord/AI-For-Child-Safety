const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static(__dirname));

app.get('/analyze/:videoId', (req, res) => {
    const videoId = req.params.videoId;
    
    const pythonProcess = spawn('python', ['app.py', videoId]);

    pythonProcess.stdout.on('data', (data) => {
        const result = data.toString().trim();
        res.json({ status: result });
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });
});

app.post('/feedback', (req, res) => {
    const { videoId, type } = req.body;
    const logEntry = `${videoId},${type}\n`;
    
    fs.appendFile('feedback.csv', logEntry, (err) => {
        if (err) return res.status(500).send("Error saving feedback");
        res.send("Feedback logged");
    });
});

app.listen(port, () => {
    console.log(`AI Monitor running at http://localhost:${port}`);
});