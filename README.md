# Importing the Dependencies


```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
```

## Data Collection and Analysis

## PIMA Diabetes Dataset


```python
# loading the diabetes dataset to a pandas Dataframe
diabetes_dataset=pd.read_csv("diabetes.csv")
```


```python
pd.read_csv?
```


    [1;31mSignature:[0m
    [0mpd[0m[1;33m.[0m[0mread_csv[0m[1;33m([0m[1;33m
    [0m    [0mfilepath_or_buffer[0m[1;33m:[0m [1;34m'FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str]'[0m[1;33m,[0m[1;33m
    [0m    [1;33m*[0m[1;33m,[0m[1;33m
    [0m    [0msep[0m[1;33m:[0m [1;34m'str | None | lib.NoDefault'[0m [1;33m=[0m [1;33m<[0m[0mno_default[0m[1;33m>[0m[1;33m,[0m[1;33m
    [0m    [0mdelimiter[0m[1;33m:[0m [1;34m'str | None | lib.NoDefault'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mheader[0m[1;33m:[0m [1;34m"int | Sequence[int] | None | Literal['infer']"[0m [1;33m=[0m [1;34m'infer'[0m[1;33m,[0m[1;33m
    [0m    [0mnames[0m[1;33m:[0m [1;34m'Sequence[Hashable] | None | lib.NoDefault'[0m [1;33m=[0m [1;33m<[0m[0mno_default[0m[1;33m>[0m[1;33m,[0m[1;33m
    [0m    [0mindex_col[0m[1;33m:[0m [1;34m'IndexLabel | Literal[False] | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0musecols[0m[1;33m:[0m [1;34m'UsecolsArgType'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mdtype[0m[1;33m:[0m [1;34m'DtypeArg | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mengine[0m[1;33m:[0m [1;34m'CSVEngine | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mconverters[0m[1;33m:[0m [1;34m'Mapping[Hashable, Callable] | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mtrue_values[0m[1;33m:[0m [1;34m'list | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mfalse_values[0m[1;33m:[0m [1;34m'list | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mskipinitialspace[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mFalse[0m[1;33m,[0m[1;33m
    [0m    [0mskiprows[0m[1;33m:[0m [1;34m'list[int] | int | Callable[[Hashable], bool] | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mskipfooter[0m[1;33m:[0m [1;34m'int'[0m [1;33m=[0m [1;36m0[0m[1;33m,[0m[1;33m
    [0m    [0mnrows[0m[1;33m:[0m [1;34m'int | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mna_values[0m[1;33m:[0m [1;34m'Hashable | Iterable[Hashable] | Mapping[Hashable, Iterable[Hashable]] | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mkeep_default_na[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mTrue[0m[1;33m,[0m[1;33m
    [0m    [0mna_filter[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mTrue[0m[1;33m,[0m[1;33m
    [0m    [0mverbose[0m[1;33m:[0m [1;34m'bool | lib.NoDefault'[0m [1;33m=[0m [1;33m<[0m[0mno_default[0m[1;33m>[0m[1;33m,[0m[1;33m
    [0m    [0mskip_blank_lines[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mTrue[0m[1;33m,[0m[1;33m
    [0m    [0mparse_dates[0m[1;33m:[0m [1;34m'bool | Sequence[Hashable] | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0minfer_datetime_format[0m[1;33m:[0m [1;34m'bool | lib.NoDefault'[0m [1;33m=[0m [1;33m<[0m[0mno_default[0m[1;33m>[0m[1;33m,[0m[1;33m
    [0m    [0mkeep_date_col[0m[1;33m:[0m [1;34m'bool | lib.NoDefault'[0m [1;33m=[0m [1;33m<[0m[0mno_default[0m[1;33m>[0m[1;33m,[0m[1;33m
    [0m    [0mdate_parser[0m[1;33m:[0m [1;34m'Callable | lib.NoDefault'[0m [1;33m=[0m [1;33m<[0m[0mno_default[0m[1;33m>[0m[1;33m,[0m[1;33m
    [0m    [0mdate_format[0m[1;33m:[0m [1;34m'str | dict[Hashable, str] | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mdayfirst[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mFalse[0m[1;33m,[0m[1;33m
    [0m    [0mcache_dates[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mTrue[0m[1;33m,[0m[1;33m
    [0m    [0miterator[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mFalse[0m[1;33m,[0m[1;33m
    [0m    [0mchunksize[0m[1;33m:[0m [1;34m'int | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mcompression[0m[1;33m:[0m [1;34m'CompressionOptions'[0m [1;33m=[0m [1;34m'infer'[0m[1;33m,[0m[1;33m
    [0m    [0mthousands[0m[1;33m:[0m [1;34m'str | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mdecimal[0m[1;33m:[0m [1;34m'str'[0m [1;33m=[0m [1;34m'.'[0m[1;33m,[0m[1;33m
    [0m    [0mlineterminator[0m[1;33m:[0m [1;34m'str | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mquotechar[0m[1;33m:[0m [1;34m'str'[0m [1;33m=[0m [1;34m'"'[0m[1;33m,[0m[1;33m
    [0m    [0mquoting[0m[1;33m:[0m [1;34m'int'[0m [1;33m=[0m [1;36m0[0m[1;33m,[0m[1;33m
    [0m    [0mdoublequote[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mTrue[0m[1;33m,[0m[1;33m
    [0m    [0mescapechar[0m[1;33m:[0m [1;34m'str | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mcomment[0m[1;33m:[0m [1;34m'str | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mencoding[0m[1;33m:[0m [1;34m'str | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mencoding_errors[0m[1;33m:[0m [1;34m'str | None'[0m [1;33m=[0m [1;34m'strict'[0m[1;33m,[0m[1;33m
    [0m    [0mdialect[0m[1;33m:[0m [1;34m'str | csv.Dialect | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mon_bad_lines[0m[1;33m:[0m [1;34m'str'[0m [1;33m=[0m [1;34m'error'[0m[1;33m,[0m[1;33m
    [0m    [0mdelim_whitespace[0m[1;33m:[0m [1;34m'bool | lib.NoDefault'[0m [1;33m=[0m [1;33m<[0m[0mno_default[0m[1;33m>[0m[1;33m,[0m[1;33m
    [0m    [0mlow_memory[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mTrue[0m[1;33m,[0m[1;33m
    [0m    [0mmemory_map[0m[1;33m:[0m [1;34m'bool'[0m [1;33m=[0m [1;32mFalse[0m[1;33m,[0m[1;33m
    [0m    [0mfloat_precision[0m[1;33m:[0m [1;34m"Literal['high', 'legacy'] | None"[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mstorage_options[0m[1;33m:[0m [1;34m'StorageOptions | None'[0m [1;33m=[0m [1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mdtype_backend[0m[1;33m:[0m [1;34m'DtypeBackend | lib.NoDefault'[0m [1;33m=[0m [1;33m<[0m[0mno_default[0m[1;33m>[0m[1;33m,[0m[1;33m
    [0m[1;33m)[0m [1;33m->[0m [1;34m'DataFrame | TextFileReader'[0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m
    Read a comma-separated values (csv) file into DataFrame.
    
    Also supports optionally iterating or breaking of the file
    into chunks.
    
    Additional help can be found in the online docs for
    `IO Tools <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.
    
    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is
        expected. A local file could be: file://localhost/path/to/table.csv.
    
        If you want to pass in a path object, pandas accepts any ``os.PathLike``.
    
        By file-like object, we refer to objects with a ``read()`` method, such as
        a file handle (e.g. via builtin ``open`` function) or ``StringIO``.
    sep : str, default ','
        Character or regex pattern to treat as the delimiter. If ``sep=None``, the
        C engine cannot automatically detect
        the separator, but the Python parsing engine can, meaning the latter will
        be used and automatically detect the separator from only the first valid
        row of the file by Python's builtin sniffer tool, ``csv.Sniffer``.
        In addition, separators longer than 1 character and different from
        ``'\s+'`` will be interpreted as regular expressions and will also force
        the use of the Python parsing engine. Note that regex delimiters are prone
        to ignoring quoted data. Regex example: ``'\r\t'``.
    delimiter : str, optional
        Alias for ``sep``.
    header : int, Sequence of int, 'infer' or None, default 'infer'
        Row number(s) containing column labels and marking the start of the
        data (zero-indexed). Default behavior is to infer the column names: if no ``names``
        are passed the behavior is identical to ``header=0`` and column
        names are inferred from the first line of the file, if column
        names are passed explicitly to ``names`` then the behavior is identical to
        ``header=None``. Explicitly pass ``header=0`` to be able to
        replace existing names. The header can be a list of integers that
        specify row locations for a :class:`~pandas.MultiIndex` on the columns
        e.g. ``[0, 1, 3]``. Intervening rows that are not specified will be
        skipped (e.g. 2 in this example is skipped). Note that this
        parameter ignores commented lines and empty lines if
        ``skip_blank_lines=True``, so ``header=0`` denotes the first line of
        data rather than the first line of the file.
    names : Sequence of Hashable, optional
        Sequence of column labels to apply. If the file contains a header row,
        then you should explicitly pass ``header=0`` to override the column names.
        Duplicates in this list are not allowed.
    index_col : Hashable, Sequence of Hashable or False, optional
      Column(s) to use as row label(s), denoted either by column labels or column
      indices.  If a sequence of labels or indices is given, :class:`~pandas.MultiIndex`
      will be formed for the row labels.
    
      Note: ``index_col=False`` can be used to force pandas to *not* use the first
      column as the index, e.g., when you have a malformed file with delimiters at
      the end of each line.
    usecols : Sequence of Hashable or Callable, optional
        Subset of columns to select, denoted either by column labels or column indices.
        If list-like, all elements must either
        be positional (i.e. integer indices into the document columns) or strings
        that correspond to column names provided either by the user in ``names`` or
        inferred from the document header row(s). If ``names`` are given, the document
        header row(s) are not taken into account. For example, a valid list-like
        ``usecols`` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
        Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.
        To instantiate a :class:`~pandas.DataFrame` from ``data`` with element order
        preserved use ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]``
        for columns in ``['foo', 'bar']`` order or
        ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``
        for ``['bar', 'foo']`` order.
    
        If callable, the callable function will be evaluated against the column
        names, returning names where the callable function evaluates to ``True``. An
        example of a valid callable argument would be ``lambda x: x.upper() in
        ['AAA', 'BBB', 'DDD']``. Using this parameter results in much faster
        parsing time and lower memory usage.
    dtype : dtype or dict of {Hashable : dtype}, optional
        Data type(s) to apply to either the whole dataset or individual columns.
        E.g., ``{'a': np.float64, 'b': np.int32, 'c': 'Int64'}``
        Use ``str`` or ``object`` together with suitable ``na_values`` settings
        to preserve and not interpret ``dtype``.
        If ``converters`` are specified, they will be applied INSTEAD
        of ``dtype`` conversion.
    
        .. versionadded:: 1.5.0
    
            Support for ``defaultdict`` was added. Specify a ``defaultdict`` as input where
            the default determines the ``dtype`` of the columns which are not explicitly
            listed.
    engine : {'c', 'python', 'pyarrow'}, optional
        Parser engine to use. The C and pyarrow engines are faster, while the python engine
        is currently more feature-complete. Multithreading is currently only supported by
        the pyarrow engine.
    
        .. versionadded:: 1.4.0
    
            The 'pyarrow' engine was added as an *experimental* engine, and some features
            are unsupported, or may not work correctly, with this engine.
    converters : dict of {Hashable : Callable}, optional
        Functions for converting values in specified columns. Keys can either
        be column labels or column indices.
    true_values : list, optional
        Values to consider as ``True`` in addition to case-insensitive variants of 'True'.
    false_values : list, optional
        Values to consider as ``False`` in addition to case-insensitive variants of 'False'.
    skipinitialspace : bool, default False
        Skip spaces after delimiter.
    skiprows : int, list of int or Callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (``int``)
        at the start of the file.
    
        If callable, the callable function will be evaluated against the row
        indices, returning ``True`` if the row should be skipped and ``False`` otherwise.
        An example of a valid callable argument would be ``lambda x: x in [0, 2]``.
    skipfooter : int, default 0
        Number of lines at bottom of file to skip (Unsupported with ``engine='c'``).
    nrows : int, optional
        Number of rows of file to read. Useful for reading pieces of large files.
    na_values : Hashable, Iterable of Hashable or dict of {Hashable : Iterable}, optional
        Additional strings to recognize as ``NA``/``NaN``. If ``dict`` passed, specific
        per-column ``NA`` values.  By default the following values are interpreted as
        ``NaN``: " ", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
        "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
        "n/a", "nan", "null ".
    
    keep_default_na : bool, default True
        Whether or not to include the default ``NaN`` values when parsing the data.
        Depending on whether ``na_values`` is passed in, the behavior is as follows:
    
        * If ``keep_default_na`` is ``True``, and ``na_values`` are specified, ``na_values``
          is appended to the default ``NaN`` values used for parsing.
        * If ``keep_default_na`` is ``True``, and ``na_values`` are not specified, only
          the default ``NaN`` values are used for parsing.
        * If ``keep_default_na`` is ``False``, and ``na_values`` are specified, only
          the ``NaN`` values specified ``na_values`` are used for parsing.
        * If ``keep_default_na`` is ``False``, and ``na_values`` are not specified, no
          strings will be parsed as ``NaN``.
    
        Note that if ``na_filter`` is passed in as ``False``, the ``keep_default_na`` and
        ``na_values`` parameters will be ignored.
    na_filter : bool, default True
        Detect missing value markers (empty strings and the value of ``na_values``). In
        data without any ``NA`` values, passing ``na_filter=False`` can improve the
        performance of reading a large file.
    verbose : bool, default False
        Indicate number of ``NA`` values placed in non-numeric columns.
    
        .. deprecated:: 2.2.0
    skip_blank_lines : bool, default True
        If ``True``, skip over blank lines rather than interpreting as ``NaN`` values.
    parse_dates : bool, list of Hashable, list of lists or dict of {Hashable : list}, default False
        The behavior is as follows:
    
        * ``bool``. If ``True`` -> try parsing the index. Note: Automatically set to
          ``True`` if ``date_format`` or ``date_parser`` arguments have been passed.
        * ``list`` of ``int`` or names. e.g. If ``[1, 2, 3]`` -> try parsing columns 1, 2, 3
          each as a separate date column.
        * ``list`` of ``list``. e.g.  If ``[[1, 3]]`` -> combine columns 1 and 3 and parse
          as a single date column. Values are joined with a space before parsing.
        * ``dict``, e.g. ``{'foo' : [1, 3]}`` -> parse columns 1, 3 as date and call
          result 'foo'. Values are joined with a space before parsing.
    
        If a column or index cannot be represented as an array of ``datetime``,
        say because of an unparsable value or a mixture of timezones, the column
        or index will be returned unaltered as an ``object`` data type. For
        non-standard ``datetime`` parsing, use :func:`~pandas.to_datetime` after
        :func:`~pandas.read_csv`.
    
        Note: A fast-path exists for iso8601-formatted dates.
    infer_datetime_format : bool, default False
        If ``True`` and ``parse_dates`` is enabled, pandas will attempt to infer the
        format of the ``datetime`` strings in the columns, and if it can be inferred,
        switch to a faster method of parsing them. In some cases this can increase
        the parsing speed by 5-10x.
    
        .. deprecated:: 2.0.0
            A strict version of this argument is now the default, passing it has no effect.
    
    keep_date_col : bool, default False
        If ``True`` and ``parse_dates`` specifies combining multiple columns then
        keep the original columns.
    date_parser : Callable, optional
        Function to use for converting a sequence of string columns to an array of
        ``datetime`` instances. The default uses ``dateutil.parser.parser`` to do the
        conversion. pandas will try to call ``date_parser`` in three different ways,
        advancing to the next if an exception occurs: 1) Pass one or more arrays
        (as defined by ``parse_dates``) as arguments; 2) concatenate (row-wise) the
        string values from the columns defined by ``parse_dates`` into a single array
        and pass that; and 3) call ``date_parser`` once for each row using one or
        more strings (corresponding to the columns defined by ``parse_dates``) as
        arguments.
    
        .. deprecated:: 2.0.0
           Use ``date_format`` instead, or read in as ``object`` and then apply
           :func:`~pandas.to_datetime` as-needed.
    date_format : str or dict of column -> format, optional
        Format to use for parsing dates when used in conjunction with ``parse_dates``.
        The strftime to parse time, e.g. :const:`"%d/%m/%Y"`. See
        `strftime documentation
        <https://docs.python.org/3/library/datetime.html
        #strftime-and-strptime-behavior>`_ for more information on choices, though
        note that :const:`"%f"` will parse all the way up to nanoseconds.
        You can also pass:
    
        - "ISO8601", to parse any `ISO8601 <https://en.wikipedia.org/wiki/ISO_8601>`_
            time string (not necessarily in exactly the same format);
        - "mixed", to infer the format for each element individually. This is risky,
            and you should probably use it along with `dayfirst`.
    
        .. versionadded:: 2.0.0
    dayfirst : bool, default False
        DD/MM format dates, international and European format.
    cache_dates : bool, default True
        If ``True``, use a cache of unique, converted dates to apply the ``datetime``
        conversion. May produce significant speed-up when parsing duplicate
        date strings, especially ones with timezone offsets.
    
    iterator : bool, default False
        Return ``TextFileReader`` object for iteration or getting chunks with
        ``get_chunk()``.
    chunksize : int, optional
        Number of lines to read from the file per chunk. Passing a value will cause the
        function to return a ``TextFileReader`` object for iteration.
        See the `IO Tools docs
        <https://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking>`_
        for more information on ``iterator`` and ``chunksize``.
    
    compression : str or dict, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer' and 'filepath_or_buffer' is
        path-like, then detect compression from the following extensions: '.gz',
        '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz' or '.tar.bz2'
        (otherwise no compression).
        If using 'zip' or 'tar', the ZIP file must contain only one data file to be read in.
        Set to ``None`` for no decompression.
        Can also be a dict with key ``'method'`` set
        to one of {``'zip'``, ``'gzip'``, ``'bz2'``, ``'zstd'``, ``'xz'``, ``'tar'``} and
        other key-value pairs are forwarded to
        ``zipfile.ZipFile``, ``gzip.GzipFile``,
        ``bz2.BZ2File``, ``zstandard.ZstdDecompressor``, ``lzma.LZMAFile`` or
        ``tarfile.TarFile``, respectively.
        As an example, the following could be passed for Zstandard decompression using a
        custom compression dictionary:
        ``compression={'method': 'zstd', 'dict_data': my_compression_dict}``.
    
        .. versionadded:: 1.5.0
            Added support for `.tar` files.
    
        .. versionchanged:: 1.4.0 Zstandard support.
    
    thousands : str (length 1), optional
        Character acting as the thousands separator in numerical values.
    decimal : str (length 1), default '.'
        Character to recognize as decimal point (e.g., use ',' for European data).
    lineterminator : str (length 1), optional
        Character used to denote a line break. Only valid with C parser.
    quotechar : str (length 1), optional
        Character used to denote the start and end of a quoted item. Quoted
        items can include the ``delimiter`` and it will be ignored.
    quoting : {0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL, 2 or csv.QUOTE_NONNUMERIC, 3 or csv.QUOTE_NONE}, default csv.QUOTE_MINIMAL
        Control field quoting behavior per ``csv.QUOTE_*`` constants. Default is
        ``csv.QUOTE_MINIMAL`` (i.e., 0) which implies that only fields containing special
        characters are quoted (e.g., characters defined in ``quotechar``, ``delimiter``,
        or ``lineterminator``.
    doublequote : bool, default True
       When ``quotechar`` is specified and ``quoting`` is not ``QUOTE_NONE``, indicate
       whether or not to interpret two consecutive ``quotechar`` elements INSIDE a
       field as a single ``quotechar`` element.
    escapechar : str (length 1), optional
        Character used to escape other characters.
    comment : str (length 1), optional
        Character indicating that the remainder of line should not be parsed.
        If found at the beginning
        of a line, the line will be ignored altogether. This parameter must be a
        single character. Like empty lines (as long as ``skip_blank_lines=True``),
        fully commented lines are ignored by the parameter ``header`` but not by
        ``skiprows``. For example, if ``comment='#'``, parsing
        ``#empty\na,b,c\n1,2,3`` with ``header=0`` will result in ``'a,b,c'`` being
        treated as the header.
    encoding : str, optional, default 'utf-8'
        Encoding to use for UTF when reading/writing (ex. ``'utf-8'``). `List of Python
        standard encodings
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .
    
    encoding_errors : str, optional, default 'strict'
        How encoding errors are treated. `List of possible values
        <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .
    
        .. versionadded:: 1.3.0
    
    dialect : str or csv.Dialect, optional
        If provided, this parameter will override values (default or not) for the
        following parameters: ``delimiter``, ``doublequote``, ``escapechar``,
        ``skipinitialspace``, ``quotechar``, and ``quoting``. If it is necessary to
        override values, a ``ParserWarning`` will be issued. See ``csv.Dialect``
        documentation for more details.
    on_bad_lines : {'error', 'warn', 'skip'} or Callable, default 'error'
        Specifies what to do upon encountering a bad line (a line with too many fields).
        Allowed values are :
    
        - ``'error'``, raise an Exception when a bad line is encountered.
        - ``'warn'``, raise a warning when a bad line is encountered and skip that line.
        - ``'skip'``, skip bad lines without raising or warning when they are encountered.
    
        .. versionadded:: 1.3.0
    
        .. versionadded:: 1.4.0
    
            - Callable, function with signature
              ``(bad_line: list[str]) -> list[str] | None`` that will process a single
              bad line. ``bad_line`` is a list of strings split by the ``sep``.
              If the function returns ``None``, the bad line will be ignored.
              If the function returns a new ``list`` of strings with more elements than
              expected, a ``ParserWarning`` will be emitted while dropping extra elements.
              Only supported when ``engine='python'``
    
        .. versionchanged:: 2.2.0
    
            - Callable, function with signature
              as described in `pyarrow documentation
              <https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html
              #pyarrow.csv.ParseOptions.invalid_row_handler>`_ when ``engine='pyarrow'``
    
    delim_whitespace : bool, default False
        Specifies whether or not whitespace (e.g. ``' '`` or ``'\t'``) will be
        used as the ``sep`` delimiter. Equivalent to setting ``sep='\s+'``. If this option
        is set to ``True``, nothing should be passed in for the ``delimiter``
        parameter.
    
        .. deprecated:: 2.2.0
            Use ``sep="\s+"`` instead.
    low_memory : bool, default True
        Internally process the file in chunks, resulting in lower memory use
        while parsing, but possibly mixed type inference.  To ensure no mixed
        types either set ``False``, or specify the type with the ``dtype`` parameter.
        Note that the entire file is read into a single :class:`~pandas.DataFrame`
        regardless, use the ``chunksize`` or ``iterator`` parameter to return the data in
        chunks. (Only valid with C parser).
    memory_map : bool, default False
        If a filepath is provided for ``filepath_or_buffer``, map the file object
        directly onto memory and access the data directly from there. Using this
        option can improve performance because there is no longer any I/O overhead.
    float_precision : {'high', 'legacy', 'round_trip'}, optional
        Specifies which converter the C engine should use for floating-point
        values. The options are ``None`` or ``'high'`` for the ordinary converter,
        ``'legacy'`` for the original lower precision pandas converter, and
        ``'round_trip'`` for the round-trip converter.
    
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc. For HTTP(S) URLs the key-value pairs
        are forwarded to ``urllib.request.Request`` as header options. For other
        URLs (e.g. starting with "s3://", and "gcs://") the key-value pairs are
        forwarded to ``fsspec.open``. Please see ``fsspec`` and ``urllib`` for more
        details, and for more examples on storage options refer `here
        <https://pandas.pydata.org/docs/user_guide/io.html?
        highlight=storage_options#reading-writing-remote-files>`_.
    
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:
    
        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.
    
        .. versionadded:: 2.0
    
    Returns
    -------
    DataFrame or TextFileReader
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.
    
    See Also
    --------
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_table : Read general delimited file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.
    
    Examples
    --------
    >>> pd.read_csv('data.csv')  # doctest: +SKIP
    [1;31mFile:[0m      c:\users\hp\appdata\local\programs\python\python313\lib\site-packages\pandas\io\parsers\readers.py
    [1;31mType:[0m      function



```python
# printing the first 5 rows of the dataset
diabetes_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# number of rows and columns in this dataset
diabetes_dataset.shape
```




    (768, 9)




```python
# getting the statistical measures of the data
diabetes_dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
diabetes_dataset['Outcome'].value_counts()
```




    Outcome
    0    500
    1    268
    Name: count, dtype: int64



0 --> Non-Diabetic
1--> Diabetic


```python
diabetes_dataset.groupby('Outcome').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Outcome</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.298000</td>
      <td>109.980000</td>
      <td>68.184000</td>
      <td>19.664000</td>
      <td>68.792000</td>
      <td>30.304200</td>
      <td>0.429734</td>
      <td>31.190000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.865672</td>
      <td>141.257463</td>
      <td>70.824627</td>
      <td>22.164179</td>
      <td>100.335821</td>
      <td>35.142537</td>
      <td>0.550500</td>
      <td>37.067164</td>
    </tr>
  </tbody>
</table>
</div>




```python
# seperating the data and labels
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']
```


```python
print(X)
```

         Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0              6      148             72             35        0  33.6   
    1              1       85             66             29        0  26.6   
    2              8      183             64              0        0  23.3   
    3              1       89             66             23       94  28.1   
    4              0      137             40             35      168  43.1   
    ..           ...      ...            ...            ...      ...   ...   
    763           10      101             76             48      180  32.9   
    764            2      122             70             27        0  36.8   
    765            5      121             72             23      112  26.2   
    766            1      126             60              0        0  30.1   
    767            1       93             70             31        0  30.4   
    
         DiabetesPedigreeFunction  Age  
    0                       0.627   50  
    1                       0.351   31  
    2                       0.672   32  
    3                       0.167   21  
    4                       2.288   33  
    ..                        ...  ...  
    763                     0.171   63  
    764                     0.340   27  
    765                     0.245   30  
    766                     0.349   47  
    767                     0.315   23  
    
    [768 rows x 8 columns]
    


```python
print(Y)
```

    0      1
    1      0
    2      1
    3      0
    4      1
          ..
    763    0
    764    0
    765    0
    766    1
    767    0
    Name: Outcome, Length: 768, dtype: int64
    

## Data Standardization


```python
scaler=StandardScaler()
```


```python
standardized_data=scaler.fit_transform(X)
```


```python
print(standardized_data)
```

    [[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
       1.4259954 ]
     [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
      -0.19067191]
     [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
      -0.10558415]
     ...
     [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
      -0.27575966]
     [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
       1.17073215]
     [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
      -0.87137393]]
    


```python
X=standardized_data
Y=diabetes_dataset['Outcome']
```


```python
print(X)
print(Y)
```

    [[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
       1.4259954 ]
     [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
      -0.19067191]
     [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
      -0.10558415]
     ...
     [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
      -0.27575966]
     [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
       1.17073215]
     [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
      -0.87137393]]
    0      1
    1      0
    2      1
    3      0
    4      1
          ..
    763    0
    764    0
    765    0
    766    1
    767    0
    Name: Outcome, Length: 768, dtype: int64
    

## Train Test Split


```python
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)
```


```python
print(X.shape,X_train.shape,X_test.shape)
```

    (768, 8) (614, 8) (154, 8)
    

## Training the Model


```python

classifier=svm.SVC(kernel='linear')
```


```python
# Training the support vector Machine Classifier
classifier.fit(X_train,Y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC(kernel=&#x27;linear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SVC</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>SVC(kernel=&#x27;linear&#x27;)</pre></div> </div></div></div></div>



## Model Evaluation

## Accuracy score 


```python
# Accuracy score on the training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
```


```python
print("Accuracy score of the training data: ",training_data_accuracy)
```

    Accuracy score of the training data:  0.7866449511400652
    


```python
# Accuracy score on the training data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
```


```python
print("Accuracy score of the test data: ",test_data_accuracy)
```

    Accuracy score of the test data:  0.7727272727272727
    

## Making a predictive system


```python
input_data=(4,110,92,0,0,37.6,0.191,30)

# Changing the input data into numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshape the array as we are predicting for one instance 
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

# strandardize the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)

prediction =classifier.predict(std_data)
print(prediction)

if prediction[0]==0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
```

    [[ 0.04601433 -0.34096773  1.18359575 -1.28821221 -0.69289057  0.71168975
      -0.84827977 -0.27575966]]
    [0]
    The person is not diabetic
    
