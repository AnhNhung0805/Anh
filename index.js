require("dotenv").config();

const express = require("express");
const app = express();
const port = 3000;

var mongoose = require("mongoose");
mongoose.connect(process.env.MONGO_URL);

var bodyParser = require("body-parser");
var userRoute = require("./routes/user.route");
var authRoute = require("./routes/auth.route");
var authMiddleware = require("./middlewares/auth.middleware");
var sessionMiddleware = require("./middlewares/session.middleware");
var productRoute = require("./routes/product.route");
var cartRoute = require("./routes/cart.route");
var apiProductRoute = require("./api/routes/product.route");

// cookie
var cookieParser = require("cookie-parser");
app.use(cookieParser(process.env.SESSION_SECRET));
// cookie

app.set("view engine", "pug");
app.set("views", "./views");

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use(express.static("public"));

app.use(sessionMiddleware);

app.get("/", (req, res) =>
    res.render("index", {
        name: "Kratos"
    })
);

app.use("/users", authMiddleware.requireAuth, userRoute); // route /user
app.use("/auth", authRoute); // route /auth
app.use("/products", productRoute);
app.use("/cart", cartRoute);
app.use("/api/products", apiProductRoute);

app.listen(port, () => console.log(`Example app listening on port ${port}!`));