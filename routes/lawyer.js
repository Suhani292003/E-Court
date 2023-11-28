const express = require('express');
const router = express.Router();
router.use(express.static('lawyer'))
const userController = require('../controllers/lawyer_controller');

router.get('/register',userController.register);
router.get('/logging',userController.logging);
router.get('/number',userController.number)
router.get('/dashboard_lawyer',userController.dashboard_lawyer);
router.get('/complaint',userController.complaint)
// router.get('/otp',userController.otp);
router.post('/lawyer_register',userController.lawyer_register);
router.post('/lawyer_login',userController.lawyer_login);
router.post('/numbers',userController.numbers);
router.post('/case',userController.case);
router.post('/complaintBox',userController.complaintBox)
module.exports = router;