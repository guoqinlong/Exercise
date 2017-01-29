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
                .json(response.message);
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

var _addReview = function(req, res, hotel) {

    hotel.reviews.push({
        name : req.body.name,
        rating : parseInt(req.body.rating, 10),
        review : req.body.review
    });

    hotel.save(function(err, hotelReturned) {
        if (err) {
            err
                .status(500)
                .json(err);
        } else {
            res
                .status(201)
                .json(hotelReturned.reviews[hotelReturned.reviews.length - 1]);
        }
    });

}

module.exports.reviewsAddOne = function(req, res) {

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
            }
            if (doc) {
                _addReview(req, res, doc);
            } else {
                res
                    .status(response.status)
                    .json(response.message);
            }
        });
}

module.exports.reviewsUpdateOne = function(req, res) {
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
            } else if (!hotel.reviews.id(reviewId)) {
                console.log("Review id not found", reviewId);
                response.status = 404;
                response.message = {
                    "message" : "Review id not fount " + reviewId
                };
            };
            if (response.status != 200) {
                res
                    .status(response.status)
                    .json(response.message);
            }
            var review = hotel.reviews.id(reviewId);
            review.name = req.body.name;
            review.rating = parseInt(req.body.rating, 10);
            review.review = req.body.review;
            hotel.save(function(err, returnedHotel) {
               if (err) {
                   res
                       .status(500)
                       .json(err);
               } else {
                   res
                       .status(204)
                       .json();
               }
            });
        });
};

module.exports.reviewsDeleteOne = function(req, res) {
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
            } else if (!hotel.reviews.id(reviewId)) {
                console.log("Review id not found", reviewId);
                response.status = 404;
                response.message = {
                    "message" : "Review id not fount " + reviewId
                };
            };
            if (response.status != 200) {
                res
                    .status(response.status)
                    .json(response.message);
            }
            hotel.reviews.id(reviewId).remove();
            hotel.save(function(err, returnedHotel) {
                if (err) {
                    res
                        .status(500)
                        .json(err);
                } else {
                    res
                        .status(204)
                        .json();
                }
            });
        });

};