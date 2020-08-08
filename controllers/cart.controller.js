var db = require("../db");
module.exports.addToCart = (req, res) => {
    var productId = req.params.productId;
    var sessionId = req.signedCookies.sessionId;

    if (!sessionId) {
        res.redirect("/products");
        return;
    }

    var count = db
        .get("session")
        .find({ id: sessionId })
        .get("cart." + productId, 0)
        .value();

    db.get("session")
        .find({ id: sessionId })
        .set("cart." + productId, count + 1)
        .write();

    // var countCart = db
    //     .get("session")
    //     .get("cart")
    //     .size()
    //     .value();
    // console.log(countCart);

    res.redirect("/products");
};