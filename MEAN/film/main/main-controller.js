angular.module('myApp').controller('MainController', MainController);

function MainController(FilmFactory) {
    var vm = this;

    FilmFactory.getAllFilms(function(response) {
        vm.films = response;
    })

    vm.name = 'Qinlong';
}

