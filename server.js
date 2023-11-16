// jshint esversion6:

const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const app = express();
const port = 3000;
app.use(bodyParser.urlencoded({extended:true}));
app.use(express.static("Public"));
app.set("view engine", "ejs");
app.set('views',path.join(__dirname,'views'));

app.get('/logging',function(req,res){
    return res.render("login",{title:"login"});
})
app.get('/',function(req,res){
    return res.render('index',{title:"E-court"});
})
app.get('/register',(req,res)=>{
    return res.render('register',{title:"register"});
})

app.listen(port, () => {
    console.log("app listening at port:",port)
})