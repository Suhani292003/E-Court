const express = require('express');
const router = express.Router();

const userController = require('../controllers/users_controller');

router.get('/register',userController.register);
router.get('/logging',userController.logging);
router.get('/dashboard_client',userController.dashboard_client);
router.post('/client_register',userController.client_register);
router.post('/client_login',userController.client_login)
// router.get('/sign-up',userController.signUp);
// router.get('/sign-in',userController.signIn);

// router.post('/create',userController.create)
module.exports = router;