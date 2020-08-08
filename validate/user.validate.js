module.exports.postCreate = (req, res, next) => {
    var err = [];
    if (!req.body.name) {
        err.push('Name is empty');
    }
    if (!req.body.phone) {
        err.push('Phone is empty');
    }
    if (err.length > 0) {
        res.render('users/create', {
            errs: err,
            values: req.body
        })
    }

    res.locals.success = true;

    next();
}