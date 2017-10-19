.. _contribute:


====================
Become a contributer
====================

Why contribute
^^^^^^^^^^^^^^

As a scientist, contributing to open source software can increase you impact on science.
The tools you help developing can be used by many researchers, doing many discoveries.
You can check out who is contributing to Atomap `here <https://gitlab.com/atomap/atomap/graphs/master>`_.

Open source software has many advantages:

1. It is better. Since the source code is open, anyone can help finding errors, improving the code and adding functionality. More about that `here <https://www.dwheeler.com/oss_fs_why.html>`_.
2. It is free.
3. It is for free.


Version control
^^^^^^^^^^^^^^^
For software development projects it is important to have a good system for keeping different versions of the software.
Atomap uses Git as version control system.
One version of Atomap is the newest released version, another is the `development version <https://gitlab.com/atomap/atomap>`_.
When new functionality is created or old functionality improved, new branches of the development version is typically made.
The branch is merged back into the master development version when they are done.

You only need to know a few basic things about git to be able to start developing.
Those are how to fork a project, `clone`

First steps
^^^^^^^^^^^

1. Create a user on `GitLab <https://gitlab.com/>`_ and `install <https://gist.github.com/derhuerst/1b15ff4652a867391f03>`_ git on your computer 
2. Make your own fork of Atomap `here <https://gitlab.com/atomap/atomap>`_ , from which you can do development. Click on the fork button. You need to be logged in.

.. figure:: images/misc/fork.jpg
    :scale: 75 %
    :align: center

3. Clone your fork to you computer, `as shown here <https://docs.gitlab.com/ce/gitlab-basics/command-line-commands.html#clone-your-project>`_. (Use something like ``git clone git@gitlab.com:username/atomap.git``, with your own username).
4. In the terminal, make a branch for your first development. For example a branch for fixing spelling errors, called FixTypos: ``git checkout -b FixTypos``. The -b is for creating a new branch. If the branch already exists write just ``git checkout FixTypos``. Too see what branches you have, write ``git branch``. Make sure you are in the correct branch.
5. Fix typos in a file.
6. Check what has been changed. In the terminal, in the Atomap folder: ``git status``. The file you have edited should listed in red.
7. Add your file to the stage for commit by ``git add filename``
8. Commit your file by ``git commit``. Write a short but clear commit message. Such as "Fixed typos in docstring".
9. Push your commit ``git push origin FixTypos``. This synchronizes the changes you have done to the remote version on gitlab.
10. When you are ready to add the branch to the Atomap development version, make a request for merging the commit into Atomap. See `here <https://docs.gitlab.com/ce/gitlab-basics/add-merge-request.html>`_.

When starting some new branch, remember to update your master fork by pulling from Atomap (upstream master).

1. Return to your master branch ``git checkout master``
2. Pull the newest changes from Atomap ``git pull upstream master``. 
3. If upstream master is not set, set it by writing ``git remote add upstream git@gitlab.com:atomap/atomap.git``
4. Check if it is there with ``git remote -v``. Upstream should be the main Atomap project. Origin Should be you fork. 

Consult the nice Gitlab `userguide <https://docs.gitlab.com/ce/gitlab-basics/README.html>`_.
Google stuff.
Or ask for help `here <https://gitlab.com/atomap/atomap/issues>`_, by adding an `issue <https://docs.gitlab.com/ce/user/project/issues/create_new_issue.html>`_. Best of luck!
