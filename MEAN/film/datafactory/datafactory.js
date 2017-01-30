angular.module('myApp').factory('FilmFactory', FilmFactory);
function FilmFactory($http) {
    return {
        getAllFilms : getAllFilms,
        getOneFilm : getOneFilm
    };

    function getAllFilms() {
        return $http.get('http://swapi-tpiros.rhcloud.com/films').then(completed).catch(failed);
    }

    function getOneFilm(id) {
        return $http.get('http://swapi-tpiros.rhcloud.com/films/' + id).then(completed).catch(failed);
    }

    function completed(response) {
        return response.data;
    }

    function failed(response) {
        return response.statusText;
    }
}