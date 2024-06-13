$(document).ready( function () {
    $('table.datatable').DataTable( {
    "paging": false,
    "layout": {
        'topStart': 'info',
        'topEnd': 'search',
        'bottomStart': null
    },
    "order": [[0, 'asc']]
} );
} );
