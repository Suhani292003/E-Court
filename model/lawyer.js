const mongoose = require('mongoose');

const contact_schema = new mongoose.Schema({
    email:{
        type:String,
        required:true,
        unique:true
    },
    name:{
        type:String,
        required:true
    },
    bar_association_no:{
        type:String,
        required:true,
        unique:true
    }
});

const bar_association_no = mongoose.model('bar_association_no',contact_schema);
module.exports = bar_association_no;