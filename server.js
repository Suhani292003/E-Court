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




// 
// app.post('/login_client',(req,res)=>{
//     const dataToSave = {
//         username :req.body.username,
//         password : req.body.password
//       }
//       Contact_client.create(dataToSave)
//       .then((result)=>{
//         console.log("result of client login:",result);
//       })
//       .catch((error)=>{
//         console.error("Error",error);
//         return;
//       })
//       return res.redirect('index1.html')
// });
// app.post('/login',(req,res)=>{
//   const dataToSave = {
//       username :req.body.username,
//       password : req.body.password
//     }
//     Contact.create(dataToSave)
//     .then((result)=>{
//       console.log("result of login lawyer:",result);
//     })
//     .catch((error)=>{
//       console.error("Error",error);
//       return;
//     })
//     return res.redirect('index2.html')
// });


app.listen(port, (err) => {
    if(err){
        console.log("Error generated:",err);
        return;
    }
    console.log("app listening at port:",port);
})