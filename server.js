const express = require('express');
const bodyParser = require('body-parser');
const port = 3000;
const db = require('./config/mongoose');
const app = express();



app.use(bodyParser.urlencoded({extended:true}));
app.use(bodyParser.json());
app.use('/',require('./routes'));
app.use(express.static("Public"));
app.use(express.static("controllers"));

app.listen(port, (err) => {
    if(err){
        console.log("Error generated:",err);
        return;
    }
    console.log("app listening at port:",port);
})