var db = require("../db");
var md5 = require("md5");
module.exports.login = (req, res) => {
    res.render("auth/login");
};

module.exports.postLogin = (req, res) => {
    var email = req.body.email;
    var password = req.body.password;
    var user = db
        .get("users")
        .find({ email: email })
        .value();
    if (!user) {
        res.render("auth/login", {
            errs: ["Wrong email !"],
            values: req.body
        });
    } else {
        var hashedPassword = md5(password);
        if (user.password == hashedPassword) {
            res.cookie("userId", user.id, {
                signed: true
            });
            res.redirect("/users");
        } else {
            res.render("auth/login", {
                errs: ["Wrong password !"],
                values: req.body
            });
        }
    }
};