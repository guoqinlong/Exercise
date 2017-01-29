var mongoose = require('mongoose');
var Hotel = mongoose.model('Hotel');

module.exports.hotelsGetAll = function(req, res) {


    var offset = 0;
    var count = 5;

    if (req.query && req.query.offset) {
        offset = parseInt(req.query.offset, 10);
    }

    if (req.query && req.query.count) {
        count = parseInt(req.query.count, 10);
    }

    Hotel
        .find()
        .skip(offset)
        .limit(count)
        .exec(function(err, hotels){
            console.log("Found hotels", hotels.length);
            res
                .json(hotels);
        });

};

module.exports.hotelsGetOne = function(req, res) {

    var hotelId = req.params.hotelId;
    console.log("GET hotelId", hotelId);

    Hotel
        .findById(hotelId)
        .exec(function(err, doc) {
            console.log("GET hotelId", hotelId);
            res
                .status(200)
                .json(doc);
        })
};

module.exports.hotelsAddOne = function(req, res) {
    var db = dbconn.get();
    var collection = db.collection('hotels');
    var newHotel;
    console.log("POST new hotel");

    if (req.body && req.body.name && req.body.stars) {
        newHotel = req.body;
        newHotel.stars = parseInt(req.body.stars, 10);
        collection.insertOne(newHotel, function(err, response) {
            console.log(response.ops);
            res
                .status(201)
                .json(response.ops);
        });
    } else {
        console.log("Data missing from body");
        res
            .status(400)
            .json({ message : "Required data missing from body"})
    }
};