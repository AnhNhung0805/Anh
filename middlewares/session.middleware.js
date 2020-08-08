var shortid = require("shortid");
var db = require("../db");
module.exports = function(req, res, next) {
    if (!req.signedCookies.sessionId) {
        var sessionId = shortid.generate();
        res.cookie("sessionId", sessionId, {
            signed: true,
            maxAge: 24 * 60 * 60 * 1000
        });

        db.get("session")
            .push({
                id: sessionId
            })
            .write();
    }

    next();
};