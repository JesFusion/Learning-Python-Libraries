import logging

print('''
======================================== Standard Print ========================================
      
This is a print statement. It has no metadata. Just Text
''')

# the following won't print to the console by default...
logging.debug("1. DEBUG: Something bad happened! Check code!")
logging.info("2. INFO: Model has finished training!")

# the following will print because of their high importance...
logging.warning(" 3. RAM usage at 91%!")
logging.error(" 4. ERROR: Couldn't find CNN_v1.pkl!")
logging.critical(" 5. CRITICAL: AWS SERVER ABOUT TO SELF-DESTRUCT!")


# ===================================== The ANATOMY of a RECORD =====================================

st_rep = logging.LogRecord(
    name = "status_report",
    level = logging.ERROR,
    pathname = __file__,
    lineno = 40,
    msg = "Manual Error Creation",
    args = (),
    exc_info = None
)

print(f'''
Timestamp: {st_rep.created}

Level Name: {st_rep.levelname}

File Path: {st_rep.pathname}

Message: {st_rep.getMessage()}
''')

