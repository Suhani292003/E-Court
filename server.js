// jshint esversion6:

const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.urlencoded({extended:true}));
app.use(express.static("Public"));
app.set("view engine", "ejs");
const path = require('path');

app.get('/', (req, res) => {
    res.sendFile('Public/index.html' , { root : __dirname});
})
app.get('/logging', (req, res)=> {
    res.sendFile('Public/login.html' , { root : __dirname});
})
app.get('/register', (req, res)=> {
    res.sendFile('Public/register.html' , { root : __dirname});
})

app.listen(3000, () => {
    console.log("app listening at port 3000")
})