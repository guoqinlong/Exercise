angular.module('myApp').factory('FilmFactory', FilmFactory);
function FilmFactory($http) {
    return {
        getAllFilms : getAllFilms,
        getOneFilm : getOneFilm
    };

    function getAllFilms() {
        return $http.get('http://swapi-tpiros.rhcloud.com/films').then(completed).catch(failed);
    }

    function getOneFilm() {
        return $http.get('http://swapi-tpiros.rhcloud.com/films/' + id).then(completed).catch(failed);
    }

    function completed() {
        return response.data;
    }

    function failed() {
        return response.statusText;
    }
}