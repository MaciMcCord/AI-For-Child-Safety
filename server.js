import express from 'express';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static(__dirname));

function extractVideoId(url) {
    const pattern = /(?:v=|\/)([0-9A-Za-z_-]{11})/;
    const match = url.match(pattern);
    return match ? match[1] : null;
}

app.get('/analyze', (req, res) => {
    const fullUrl = req.query.url;
    const videoId = extractVideoId(fullUrl);

    if (!videoId) {
        return res.status(400).json({ error: 'Invalid YouTube URL' });
    }

    const pythonProcess = spawn('python', ['Testing.py', videoId]);

    pythonProcess.stdout.on('data', (data) => {
        const result = data.toString().trim();
        res.json({ status: result });
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });
});

app.listen(port, () => {
    console.log(`AI Monitor running at http://localhost:${port}`);
});