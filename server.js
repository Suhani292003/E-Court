const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const port = 3000;
const db = require('./config/mongoose');
const Contact = require('./model/login');
const Register = require('./model/register')
const Register_client = require('./model/register1');
const Contact_client = require('./model/login_client');
const app = express();



app.use(bodyParser.urlencoded({extended:true}));
app.use(bodyParser.json());
app.use(express.static("Public"));



app.get('/', (req, res) => {
    res.sendFile('Public/index.html' , { root : __dirname});
})
app.get('/home', (req, res) => {
  res.sendFile('Public/index1.html' , { root : __dirname});
})
app.get('/home1', (req, res) => {
  res.sendFile('Public/index2.html' , { root : __dirname});
})

app.get('/logging', (req, res)=> {
    res.sendFile('Public/login.html' , { root : __dirname});
})
app.get('/logging1', (req, res)=> {
  res.sendFile('Public/login_client.html' , { root : __dirname});
})
app.get('/register', (req, res)=> {
    res.sendFile('Public/register.html' , { root : __dirname});
})
app.get('/register1', (req, res)=> {
  res.sendFile('Public/register_client.html' , { root : __dirname});
})
app.get('/service',(req,res)=>{
  res.sendFile('Public/services.html',{root:__dirname});
})
app.get('/about',(req,res)=>{
  res.sendFile('Public/aboutUs.html',{root:__dirname});
})
app.get('/dashboard',(req,res)=>{
  res.sendFile('Public/dashboard.html',{root:__dirname});
})
app.get('/dashboard_client',(req,res)=>{
  res.sendFile('Public/dashboard_client.html',{root:__dirname});
})
app.get('/dashboard_lawyer',(req,res)=>{
  res.sendFile('Public/dashboard_lawyer.html',{root:__dirname});
})
app.get('/case',(req,res)=>{
  res.sendFile('/Public/case.html',{root:__dirname});
})
app.get('/complaint',(req,res)=>{
  res.sendFile('Public/complaint.html',{root:__dirname});
})

app.post('/login_client',(req,res)=>{
    const dataToSave = {
        username :req.body.username,
        password : req.body.password
      }
      Contact_client.create(dataToSave)
      .then((result)=>{
        console.log("result of client login:",result);
      })
      .catch((error)=>{
        console.error("Error",error);
        return;
      })
      return res.redirect('index1.html')
});
app.post('/login',(req,res)=>{
  const dataToSave = {
      username :req.body.username,
      password : req.body.password
    }
    Contact.create(dataToSave)
    .then((result)=>{
      console.log("result of login lawyer:",result);
    })
    .catch((error)=>{
      console.error("Error",error);
      return;
    })
    return res.redirect('index2.html')
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
        console.log("result of lawyer:",result);
      })
      .catch((error)=>{
        console.error("Error",error);
        return;
      })
      return res.redirect('/login.html')
});
app.post('/register1',(req,res)=>{
  const dataToSave = {
      name:req.body.name,
      dob:req.body.dob,
      email:req.body.email,
      phone:req.body.phone,
      gender:req.body.gender,
      state:req.body.state,
      bar_association_no:req.body.bar_association_no,
      password:req.body.password,
      confirm_password:req.body.confirm_password,
      face_id:req.body.face_id
    }
    Register_client.create(dataToSave)
    .then((result)=>{
      console.log("result of register client:",result);
    })
    .catch((error)=>{
      console.error("Error",error);
      return;
    })
    return res.redirect('/login_client.html')
});

app.listen(port, (err) => {
    if(err){
        console.log("Error generated:",err);
        return;
    }
    console.log("app listening at port:",port);
})