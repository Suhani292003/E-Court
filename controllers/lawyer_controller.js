const path = require('path');
const Register = require('../model/register');
const Lawyer = require('../model/lawyer');
const CaseForm = require('../model/case');
const Complaint = require('../model/complaint')
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
module.exports.complaint = (req,res)=>{
  const filePath = path.join(__dirname, 'lawyer', 'complaint.html'); // Path to your index.html file
    res.sendFile(filePath);
}

module.exports.numbers = (req,res)=>{
  const dataToSave =  {
    name:req.body.name,
    dob:req.body.dob,
    gender:req.body.gender,
    state:req.body.state,
    bar_association_no:req.body.bar_association_no
}
Lawyer.create(dataToSave)
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
  const {email,bar_association_no} = req.body;

  try {
    // Check if the user is already registered as a lawyer
    const existingLawyer = await Register.findOne({ email }).exec();
    const existingbarNo = await Lawyer.findOne({ bar_association_no }).exec();
    const lower = req.body.name.toLowerCase()
    console.log(lower)
    console.log(existingbarNo.name)
    console.log(req.body.name)
    console.log(existingbarNo.bar_association_no)
    if (!existingLawyer && existingbarNo.bar_association_no===req.body.bar_association_no && existingbarNo.name===lower) {
      Register.create(dataToSave)
      return res.redirect('/lawyer/logging');
    } 
    if(existingbarNo.bar_association_no!==req.body.bar_association_no || existingbarNo.name!==lower){
      //   // Bar_no does not exist, show bar no not registered message
      console.log("no not register or register name is not correct")
      return res.redirect('/lawyer/register')
    }
    else {
      // User is already registered as a lawyer
      console.log("already")
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

module.exports.case = (req,res)=>{
  const dataToSave = {
    district:req.body.district,
    court:req.body.court,
    matter:req.body.matter,
    case_type:req.body.case_type,
    mact:req.body.mact,
  }
  CaseForm.create(dataToSave)
  return res.redirect('/lawyer/complaint')
}
module.exports.complaintBox = (req,res)=>{
  const dataToSave = {
    Complainant_name:req.body.Complainant_name,
    phon:req.body.phone,
    com_relation:req.body.com_relation,
    address:req.body.address,
    relative_name:req.body.relative_name,
    state:req.body.state,
    dob:req.body.dob,
    comp_district:req.body.comp_district,
    comp_age:req.body.comp_age,
    comp_village:req.body.comp_village,
    gender:req.body.gender,
    ward:req.body.ward,
    caste:req.body.caste,
    pincode:req.body.pincode
  }
  Complaint.create(dataToSave)
  return res.redirect('back');
}
