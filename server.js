// jshint esversion6:

const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const port = 3000;
const db = require('./config/mongoose');
const Contact = require('./model/login');
const Register = require('./model/register')
const app = express();

//app.set("view engine", "ejs");
//app.set('views',path.join(__dirname,'views'));

app.use(bodyParser.urlencoded({extended:true}));
app.use(express.static("Public"));



app.get('/', (req, res) => {
    res.sendFile('Public/index.html' , { root : __dirname});
})
app.get('/logging', (req, res)=> {
    res.sendFile('Public/login.html' , { root : __dirname});
})
app.get('/register', (req, res)=> {
    res.sendFile('Public/register.html' , { root : __dirname});
})

app.post('/login',(req,res)=>{
    const dataToSave = {
        username :req.body.username,
        password : req.body.password
      }
      Contact.create(dataToSave)
      .then((result)=>{
        console.log("result",result);
      })
      .catch((error)=>{
        console.error("Error",error);
        return;
      })
      return res.redirect('back')
});
app.post('/register',(req,res)=>{
    const dataToSave = {
        name:req.body.name,
        dob:req.body.dob,
        email:req.body.email,
        phone:req.body.phone,
        gender:req.body.gender,
        state:req.body.state,
        password:req.body.password,
        confirm_password:req.body.confirm_password,
        face_id:req.body.face_id
      }
      Register.create(dataToSave)
      .then((result)=>{
        console.log("result",result);
      })
      .catch((error)=>{
        console.error("Error",error);
        return;
      })
      return res.redirect('back')
});

// app.get('/logging',function(req,res){
//     return res.render("login",{title:"login"});
// })
// app.get('/',function(req,res){
//     return res.render('index',{title:"E-court"});
// })
// app.get('/register',(req,res)=>{
//     return res.render('register',{title:"register"});
// })

app.listen(port, (err) => {
    if(err){
        console.log("Error generated:",err);
        return;
    }
    console.log("app listening at port:",port);
})