const path = require('path');
const Register = require('../model/register');
const bar_association_no = require('../model/lawyer');
const Contact = require('../model/login')
module.exports.register = function(req,res){
    
    const filePath = path.join(__dirname, 'lawyer', 'register.html'); // Path to your index.html file
    res.sendFile(filePath);
}
module.exports.logging = (req,res)=>{
    const filePath = path.join(__dirname, 'lawyer', 'login.html'); // Path to your index.html file
    res.sendFile(filePath);
}
module.exports.number = (req,res)=>{
  const filePath = path.join(__dirname, 'lawyer', 'bar_no.html'); // Path to your index.html file
  res.sendFile(filePath);
}
module.exports.dashboard_lawyer = (req,res)=>{
  const filePath = path.join(__dirname, 'lawyer', 'dashboard_lawyer.html'); // Path to your index.html file
  res.sendFile(filePath);
}

module.exports.numbers = (req,res)=>{
  const dataToSave =  {
    email:req.body.email,
    name:req.body.name,
    bar_association_no:req.body.bar_association_no
}
bar_association_no.create(dataToSave)
return res.redirect('back');
}

module.exports.lawyer_register =(req,res)=>{
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
    Register.create(dataToSave)
    return res.redirect('back');
}
// module.exports.lawyer_login =(req,res)=>{
  
// }