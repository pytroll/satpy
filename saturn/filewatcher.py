#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Watch files coming in a given directory.
"""

from multiprocessing import Process
import datetime
import glob
import time
import os
import logging
import signal
import sys
ZERO = datetime.timedelta(seconds=0)
LOG = logging.getLogger("filewatcher")

class FileWatcher(Process):
    """Looks for new files, and queues them.
    """

    def __init__(self, filename_template, file_queue, frequency):
        """Looks for new files arriving at the given *frequency*, and queues
        them.
        """
        Process.__init__(self)
        self.queue = file_queue
        self.template = filename_template
        self.frequency = datetime.timedelta(minutes=frequency)
        self.running = True

    def run(self):
        """Run the file watcher.
        """

        previous_file = ""

        def stop(*args):
            """Stops a running process.
            """
            del args
            self.running = False
            
        signal.signal(signal.SIGTERM, stop)

        while self.running:
            while True and self.running:
                filelist = glob.glob(self.template)
                filelist.sort()

                try:
                    filelist = filelist[filelist.index(previous_file) + 1:]
                except (IndexError, ValueError):
                    pass

                if(len(filelist) != 0 and filelist[-1] != previous_file):
                    sleep_time = 8
                    break

                LOG.info("no new file has come, waiting %s secs"
                         %str(sleep_time))
                time.sleep(sleep_time)
                if sleep_time < 60:
                    sleep_time *= 2
                

            for i in filelist:
                LOG.debug("queueing %s..."%i)
                self.queue.put(i)
                previous_file = i
                
            if previous_file:
                last_stat = os.stat(previous_file)

                since_creation = datetime.timedelta(seconds=time.time() -
                                                    last_stat.st_ctime)

                to_wait = self.frequency - since_creation
            elif self.running:
                to_wait = datetime.timedelta(seconds=0)
            else:
                to_wait = ZERO
            if to_wait > ZERO:
                LOG.info("Waiting at least "+str(to_wait)+" for next file")
                time.sleep(to_wait.seconds + to_wait.microseconds / 1000000.0)


class FileProcessor(Process):
    """Execute *fun* on filenames provided by from *file_queue*.
    """
    def __init__(self, file_queue, fun):
        Process.__init__(self)
        self.queue = file_queue
        self.fun = fun
        self.running = True
        
    def run(self):
        """Execute the given function on files from the file queue.
        """
        def stop(*args):
            """Stops a running process.
            """
            del args
            self.running = False
            sys.exit()
        signal.signal(signal.SIGTERM, stop)

        while self.running:
            filename = self.queue.get()
            LOG.debug("processing %s"%filename)
            try:
                self.fun(filename)
            except:
                LOG.exception("Something wrong happened in %s for %s. Skipping."
                              %(str(self.fun), filename))

    def stop(self):
        """Stops a running process.
        """
        self.running = False



