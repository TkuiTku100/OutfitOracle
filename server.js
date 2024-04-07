// First, import the necessary modules
const express = require('express');
const multer = require('multer');
const fs = require('fs');
const cors = require('cors');
const path = require('path');


// Initialize the Express application
const app = express();
app.use(cors());

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function(req, file, cb) {
        cb(null, 'uploads/'); // Specify the directory to save files
    },
    filename: function(req, file, cb) {
        // Generate the file name with its original extension
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage: storage });
app.use('/uploads', express.static('uploads'));



// Define a route for the root URL
app.get('/', (req, res) => {
    res.send('Welcome to the Outfit Oracle!');
});

app.post('/add-clothing-item', upload.single('clothingImage'), (req, res) => {
    const { userName, category, color, pattern, length, fit } = req.body;
    const filePath = `wardrobe_${userName}.csv`;

    // Check if the CSV file already exists
    fs.access(filePath, fs.constants.F_OK, (existErr) => {
        let csvContent;
        if (existErr) {
            // File doesn't exist, include header
            csvContent = `Category,Color,Pattern,Length,Fit,ImagePath\n`;
        } else {
            // File exists, don't include header
            csvContent = '';
        }

        // Append the new data, including the path to the uploaded image (if any)
        const imageData = req.file ? `${req.file.filename}` : 'No Image';
        csvContent += `${category},${color},${pattern},${length},${fit},${imageData}\n`;

        // Write or append to the CSV file
        fs.writeFile(filePath, csvContent, { flag: 'a' }, (writeErr) => {
            if (writeErr) {
                console.error('Error writing CSV:', writeErr);
                return res.status(500).send('Error processing data');
            }
            res.send('Data saved successfully');
        });
    });
});


app.get('/images', (req, res) => {
    const uploadsDir = path.join(__dirname, 'uploads');

    fs.readdir(uploadsDir, (err, files) => {
        if (err) {
            console.log("Failed to list contents of directory: " + err);
            res.status(500).send('Unable to list images');
        } else {
            // Optionally filter for image files only
            const imageFiles = files.filter(file => file.match(/\.(jpg|jpeg|png|gif)$/));
            res.json(imageFiles);
        }
    });
});

// Start the server
const port = 3001;
app.listen(port, () => console.log(`Server listening on port ${port}`));