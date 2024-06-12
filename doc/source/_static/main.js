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

    $('table.area-table').DataTable( {
    "paging": true,
    "pageLength": 15,
    "dom": 'lfitp'
} );
} );
