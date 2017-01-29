var mongoose = require('mongoose');
var Hotel = mongoose.model('Hotel');

// Get all reviews for a hotel
module.exports.reviewsGetAll = function(req, res) {
    var hotelId = req.params.hotelId;
    console.log("GET hodelId", hotelId);

    Hotel
        .findById(hotelId)
        .select("reviews")
        .exec(function(err, doc){
            var response = {
                status : 200,
                message : []
            };
            if (err) {
                console.log("Error finding hotel", hotelId);
                response.status = 500;
                response.message = err;
            } else if (!doc) {
                console.log("Hotel id not found in database", hotelId);
                response.status = 404;
                response.message = {
                    "message" : "Hotel id not found " + hotelId
                };
            } else {
                response.message = doc.reviews ? doc.reviews : [];
            }
            res
                .status(response.status)
                .json(response.reviews);
        });
};

// Get single review for a hotel
module.exports.reviewsGetOne = function(req, res) {
    var hotelId = req.params.hotelId;
    var reviewId = req.params.reviewId;
    console.log("Get reviewId", reviewId, "for hotelId", hotelId);
    Hotel
        .findById(hotelId)
        .select("reviews")
        .exec(function(err, hotel) {
            var response = {
                status : 200,
                message : {}
            };
            if (err) {
                console.log("Error finding hotel", hotelId);
                response.status = 500;
                response.message = err;
            } else if (!hotel) {
                console.log('Hotel id not found in database', hotelId);
                response.status = 404;
                response.message = {
                    "message" : "Hotel id not fount in database" + hotelId
                }
            } else {
                response.message = hotel.reviews.id(reviewId);
                if (!response.message) {
                    console.log("Review id not found", reviewId);
                    response.status = 404;
                    response.message = {
                        "message" : "Review id not fount " + reviewId
                    };
                }
            }
            res
                .status(response.status)
                .json(response.message);

        });
};