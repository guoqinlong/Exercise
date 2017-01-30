angular.module('myApp', ['ngRoute']).config(config);

function config($routeProvider) {
    $routeProvider.when('/', {
        templateUrl : "main/main.html",
        controller : "MainController",
        controllerAs : "vm"
    }).when('/film/:id', {
        templateUrl : "film/film.html",
        controller : "FilmController",
        controllerAs : "vm"
    }).otherwise({
        redirectTo : '/'
    });
};
