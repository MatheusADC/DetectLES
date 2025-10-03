const express = require("express");
const tf = require("@tensorflow/tfjs"); 
const jpeg = require("jpeg-js");
const path = require("path");
const sqlite3 = require("sqlite3").verbose();
const { v4: uuidv4 } = require("uuid");
const fs = require("fs"); 
const moment = require('moment-timezone');
const app = express();
const port = 3000;

app.use(express.json({ limit: "10mb" })); 

const modelPath = path.join(__dirname, "modelo-lupus2");
const uploadsDir = path.join(__dirname, "uploads"); 

if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}

let model;

const db = new sqlite3.Database("imagens.db", (err) => {
    if (err) {
        console.error("Erro ao conectar ao banco de dados:", err.message);
    } else {
        console.log("Conectado ao banco de dados SQLite.");
        db.run(`
            CREATE TABLE IF NOT EXISTS imagens (
                id TEXT PRIMARY KEY,
                base64 TEXT NOT NULL,
                data_hora TEXT NOT NULL,
                file_path TEXT NOT NULL
            )
        `);
    }
});

function decodeImageToTensor(buffer) {
    const pixels = jpeg.decode(buffer, true);
    const { width, height, data } = pixels;

    const imgTensor = tf.tensor3d(data, [height, width, 4], "int32");
    const rgbTensor = imgTensor.slice([0, 0, 0], [-1, -1, 3]);
    return rgbTensor.expandDims(0); 
}

// Modelo será carregado no frontend, não no backend
console.log("Modelo será carregado no frontend");

app.use(express.static(path.join(__dirname, "public")));
app.use("/modelo-lupus2", express.static(modelPath));

app.use('/uploads', express.static(uploadsDir));

app.post("/classify", async (req, res) => {
    const { image } = req.body;

    if (!image) {
        return res.status(400).json({ error: "No image provided" });
    }

    try {
        const base64Data = image.replace(/^data:image\/\w+;base64,/, ""); 
        const imageBuffer = Buffer.from(base64Data, "base64");

        const id = uuidv4();
        const dataHora = moment().tz("America/Sao_Paulo").format("YYYY-MM-DD HH:mm:ss");

        const imagePath = path.join(uploadsDir, `${id}.jpg`);
        fs.writeFileSync(imagePath, imageBuffer);

        db.run(
            "INSERT INTO imagens (id, base64, data_hora, file_path) VALUES (?, ?, ?, ?)",
            [id, base64Data, dataHora, imagePath],
            (err) => {
                if (err) {
                    console.error("Erro ao inserir no banco de dados:", err.message);
                } else {
                    console.log(`Imagem salva no banco com ID: ${id}`);
                }
            }
        );

        // Apenas salvar a imagem, a classificação será feita no frontend
        res.json({
            success: true,
            id,
            dataHora,
            imagePath: `/uploads/${id}.jpg`,
            message: "Imagem salva com sucesso"
        });
    } catch (error) {
        console.error("Error classifying image:", error);
        res.status(500).json({ error: "Error processing the image" });
    }
});


app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public/html", "index.html"));
});

app.listen(port, () => {
    console.log(`Servidor rodando em http://localhost:${port}`);
});
