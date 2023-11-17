const mongoose = require('mongoose');
mongoose.connect('mongodb://localhost/E_portal')
const db = mongoose.connection;
db.on('error',console.error.bind(console,'error'));
db.once('open',()=>{
    console.log('successfully connected');
})