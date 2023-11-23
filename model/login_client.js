const mongoose = require('mongoose');

const contact_schema_client = new mongoose.Schema({
    username:{
        type:String,
        required:true
    },
    password:{
        type:String,
        required: true
    }
});

const Contact_client = mongoose.model('Contact_client',contact_schema_client);
module.exports = Contact_client;