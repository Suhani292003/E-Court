const path = require('path');
const Register = require('../model/register');
const Lawyer = require('../model/lawyer');
const Otp = require('../model/otp');
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
bar_association_no.create(dataToSave)
return res.redirect('back');
}

module.exports.lawyer_register = async (req, res) => {
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
  const {email} = req.body;

  try {
    // Check if the user is already registered as a lawyer
    const existingLawyer = await Register.findOne({ email }).exec();
    const allLawyers = await Lawyer.find({}).exec();
    const allBarNos = allLawyers.map(lawyer => lawyer.bar_association_no)
    console.log(allBarNos)
    console.log(req.body.bar_association_no)
    console.log(allBarNos.includes(req.body.bar_association_no))
    if (!existingLawyer && allBarNos.includes(req.body.bar_association_no)) {
      Register.create(dataToSave)
      // const filePath = path.join(__dirname, 'lawyer', 'login.html'); // Path to your index.html file
      // res.sendFile(filePath);
      // return 
      return res.redirect('/lawyer/logging');
    } 
    if(!allBarNos.includes(req.body.bar_association_no)){
      //   // Bar_no does not exist, show bar no not registered message
      // const filePath = path.join(__dirname, 'lawyer', 'register.html'); // Path to your index.html file
      // res.sendFile(filePath);
      // return ;
      return res.redirect('/lawyer/register')
    }
    else {
      // User is already registered as a lawyer
      // const filePath = path.join(__dirname, 'lawyer', 'register.html'); // Path to your index.html file
      // res.sendFile(filePath);
      // return ;
      return res.redirect('/lawyer/register')
    }
  } catch (error) {
    return res.status(500).json({ message: 'Internal server error', error: error.message });
  }
};


module.exports.lawyer_login =(req,res)=>{
  const submittedEmail  = req.body
  Register.findOne({email: submittedEmail.email})
  .then(existingUser=>{
    if(existingUser){
      if(existingUser.password === submittedEmail.password){
        res.cookie('lawyer_id',existingUser.id);
        res.redirect('/lawyer/dashboard_lawyer')
        console.log('join')
        return
      }else{
        res.redirect('/lawyer/logging')
        console.log("wrong pass")
        return;
        // return res.status(500).json({ message: 'Wrong password:'});
      }
    }
    else{
      res.redirect('/lawyer/logging')
      console.log('not register')
      return;
      }
    })
    .catch(error=>{
      return res.status(500).json({ message: 'Internal server error', error: error.message });
    }) 
}
