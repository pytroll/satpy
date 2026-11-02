"""Module for autogenerating reader table from config files."""

from yaml import BaseLoader

from satpy.readers.core.config import available_readers


def rst_table_row(columns=None):
    """Create one row for a rst table.

    Args:
        columns (list[str]): Content of each column.

    Returns:
        str
    """
    row = "    * - {}\n".format(columns[0])
    columns = ["      - {}\n".format(col) for col in columns[1:]]
    row = row + "".join(columns)

    return row


def rst_table_header(name=None, header=None, header_rows=1, widths="auto", class_name="datatable"):
    """Create header for rst table.

    Args:
        name (str): Name of the table
        header (list[str]): Column names
        header_rows (int): Number of header rows
        widths (optional[list[int]]): Width of each column as a list. If not specified
            defaults to auto and will therefore determined by the backend
            (see <https://docutils.sourceforge.io/docs/ref/rst/directives.html#table>)
        class_name (str): The CSS class name for the table. A corresponding js function should be in main.js in
            in the "statis" directory.

    Returns:
        str
    """
    if isinstance(widths, list):
        widths = " ".join([str(w) for w in widths])

    header = rst_table_row(header)

    table_header = (f".. list-table:: {name}\n"
                    f"    :header-rows: {header_rows}\n"
                    f"    :widths: {widths}\n"
                    f"    :class: {class_name}\n\n"
                    f"{header}")

    return table_header


def generate_reader_table():
    """Create reader table from reader yaml config files.

    Returns:
        str
    """
    table = [rst_table_header("Satpy Readers", header=["Description", "Reader name", "Status", "fsspec support"],
                              widths="auto")]

    reader_configs = available_readers(as_dict=True, yaml_loader=BaseLoader)
    for rc in reader_configs:
        table.append(rst_table_row([rc.get("long_name", "").rstrip("\n"), rc.get("name", ""),
                                    rc.get("status", ""), rc.get("supports_fsspec", "false")]))

    return "".join(table)
