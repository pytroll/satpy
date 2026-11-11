$(document).ready( function () {
    $('table.datatable').DataTable( {
    "paging": true,
    "pageLength": 15,
    "layout": {
        'topStart': 'info',
        'topEnd': 'search',
        'bottomStart': null,
        'bottomEnd': 'paging'
    },
    "order": [[0, 'asc']]
} );

    $('table.area-table').DataTable( {
    "paging": true,
    "pageLength": 15,
    "layout": {
        'topStart': 'info',
        'topEnd': 'search',
        'bottomEnd': 'paging',
        'bottomStart': null
    }
} );
} );
