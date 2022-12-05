$(document).ready( function () {
    $('table.datatable').DataTable( {
    "paging": false,
    "dom": 'lfitp'
} );

    $('table.area-table').DataTable( {
    "paging": true,
    "pageLength": 15,
    "dom": 'lfitp'
} );
} );
