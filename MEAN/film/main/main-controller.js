angular.module('myApp').controller('MainController', MainController);

function MainController(FilmFactory) {
    var vm = this;
    FilmFactory.getAllFilms().then(function(response) {
        vm.films = response;
    });
    vm.name = 'Qinlong';
    vm.date1 = '12 February 2016';
    vm.date2 = '11 Match 2016';
    vm.date3 = '03 January 2015';
    vm.date4 = '25 April 2014';
}