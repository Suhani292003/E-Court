const mongoose = require('mongoose');

const contact_schema = new mongoose.Schema({
    username:{
        type:String,
        required:true
    },
    password:{
        type:String,
        required: true
    }
});

const Contact = mongoose.model('Contact',contact_schema);
module.exports = Contact;