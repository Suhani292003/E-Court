const mongoose = require('mongoose');

const contact_schema = new mongoose.Schema({
    name:{
        type:String,
        required:true
    },
    dob:{
        type:Date,
        required: true
    },
    email:{
        type:String,
        required:true,
        unique:true
    },
    phone:{
        type:String,
        required:true
    },
    gender:{
        type:String,
        required:true
    },
    state:{
        type:String,
        required:true
    },
    bar_association_no:{
        type:String,
        required:true,
        // unique:true
    },
    password:{
        type:String,
        required:true
    },
    confirm_password:{
        type:String,
        required:true
    },
    face_id:{
        type:String,
        // required:true
    }
});

const bar_association_no = mongoose.model('bar_association_no',contact_schema);
module.exports = bar_association_no;