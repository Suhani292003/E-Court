const path = require('path');
module.exports.home = function(req,res){
    const filePath = path.join(__dirname,'index.html'); // Path to your index.html file
    res.sendFile(filePath);
}
module.exports.aboutUS = function(req,res){
    const filePath = path.join(__dirname,'aboutUs.html'); // Path to your index.html file
    res.sendFile(filePath);
}
module.exports.services = function(req,res){
    const filePath = path.join(__dirname,'services.html'); // Path to your index.html file
    res.sendFile(filePath);
}

module.exports.dashboard = function(req,res){
    const filePath = path.join(__dirname,'dashboard.html'); // Path to your index.html file
    res.sendFile(filePath);
}

module.exports.contactUs = function(req,res){
    const filePath = path.join(__dirname,'contactUs.html'); // Path to your index.html file
    res.sendFile(filePath);
}