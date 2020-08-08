var db = require("../db");
var shortid = require("shortid");

module.exports.index = (req, res) => {
    res.render("users/index", {
        users: db.get("users").value()
    });
};

module.exports.search = (req, res) => {
    var q = req.query.name;
    var matchedUsers = db
        .get("users")
        .value()
        .filter(function(user) {
            return user.name.includes(q) == 1;
        });

    res.render("users/index", {
        users: matchedUsers
    });
};

module.exports.getId = function(req, res) {
    var id = req.params.id; // take id from path

    var user = db
        .get("users")
        .find({ id: id })
        .value(); // find user have id

    res.render("users/view", {
        user: user
    });
};

module.exports.create = (req, res) => {
    res.render("users/create");
};

module.exports.postCreate = (req, res) => {
    req.body.id = shortid.generate();
    req.body.avatar = req.file.path
        .split("\\")
        .slice(1)
        .join("/");

    db.get("users")
        .push(req.body)
        .write();
    res.redirect("/users");
};