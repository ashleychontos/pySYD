.. role::  raw-html(raw)
    :format: html

.. |smiley| unicode:: U+263A

.. _community:

************
Contributing
************


.. link-button:: guidelines/index
    :type: ref
    :text: Jump to our community guidelines
    :classes: btn-outline-secondary btn-block

.. _community/contributors:

The ``pySYD`` team
##################

Our community continues to grow! See below to find out how you can help |smiley|


Contributors
************

.. include:: CONTRIBUTORS.rst

.. important::

    ``pySYD`` was initially the `Python` translation of the `IDL`-based asteroseismology 
    pipeline ``SYD``, which was written by my PhD advisor, Dan Huber, during his PhD in 
    Sydney (hence the name). Therefore none of this would have been possible without his 
    i) years of hard work during his PhD as well as ii) years of patience during my PhD |smiley|
    
    **~A very special shoutout to Dan~**

.. _community/collaborators:

Collaborators
*************

We have many amazing collaborators that have helped with the development of the software, especially with the 
improvements that have been implemented -- which have ultimately made ``pySYD`` more user-friendly. Many
thanks to our collaborators!

.. include:: COLLABORATORS.rst

-----

.. _community/guidelines:

Community guidelines
####################

For most (if not all) questions/concerns, checking our `discussions <https://github.com/ashleychontos/pySYD/discussions>`_ 
forum is a great place to start in case things have already been brought up and/or addressed.

If you would like to contribute, here are the guidelines we ask you to follow:

* :ref:`Question or problem <guidelines/question>`
* :ref:`Issues & bugs <guidelines/issue>`
* :ref:`New features <guidelines/feature>`
* :ref:`Contributing code <guidelines/contribute>`
* :ref:`Style guide <guidelines/style>`
* :ref:`Testing <guidelines/testing>`

-----

.. _community/guidelines/question:

Question or problem
*******************

:raw-html:`&rightarrow;` Do you have a general question that is not directly related to software functionality?

Please visit our relevant `thread <https://github.com/ashleychontos/pySYD/discussions/37#discussion-3918112>`_ first to see if your question has already been asked. You can also help us keep this space up-to-date, linking topics/issues to relevant threads and adding appropriate tags whenever/wherever possible. This is not only helpful to us but also helpful for the community! Once we have enough data points, we will establish a forum for frequently asked questions (FAQ).

.. warning::

    **Please do not open issues for general support questions as we want to preserve them for bug reports** 
    **and new feature requests ONLY.** Therefore to save everyone time, we will be systematically closing all 
    issues that do not follow these guidelines.

If this still does not work for you and you would like to chat with someone in real-time, please contact `Ashley <mailto:achontos@hawaii.edu>`_ to set up a chat or zoom meeting.

-----

.. _community/guidelines/issue:

Issues & bugs
*************

:raw-html:`&rightarrow;` Are you reporting a bug?

**If the code crashes or you find a bug, please search the issue tracker first to make sure the problem (i.e. issue) does not already exist.** If and only if you do this but still don't find anything, feel free to submit an issue. And, if you're *really* brave, you can submit an issue along with a pull request fix.

Ideally we would love to resolve all issues immediately but before fixing a bug, we first 
to need reproduce and confirm it. There is a template of the required information when you 
submit an issue, but generally we ask that you:

* clearly and concisely explain the issue or bug
* provide any relevant data so that we can reproduce the error
* information on the software and operating system

You can file new issues by filling out our `bug report <https://github.com/ashleychontos/pySYD/issues/new?assignees=&labels=&template=bug_report.md>`_ template.

-----

.. _community/guidelines/feature:

New features
************

:raw-html:`&rightarrow;` Have an idea for a new feature or functionality?

Request 
=======

If you come up with an idea for a new feature that you'd like to see implemented in 
``pySYD`` but do not plan to do this yourself, you can submit an issue with our 
`feature request <https://github.com/ashleychontos/pySYD/issues/new?assignees=&labels=&template=feature_request.md>`_ template.

We welcome any and all ideas!

Direct implementation
=====================

However, if you come up with a brilliant idea that you'd like to take a stab at -- 
Please first consider what kind of change it is:

* For a **Major Feature**, first open an issue and outline your proposal so that it can be
  discussed. This will also allow us to better coordinate our efforts, prevent duplication of work,
  and help you to craft the change so that it is successfully accepted into the project.
* Any smaller or **Minor Features** can be crafted and directly submitted as a pull request. However,
  before you submit a pull request, please see our :ref:`style guide <guidelines/style>` to facilitate
  and expedite the merge process.

-----

.. _community/guidelines/code:

Contributing code
*****************

:raw-html:`&rightarrow;` Do you want to contribute code?

We would love for you to contribute to ``pySYD`` and make it even better than it is today! 

Style guide
===========

** A good rule of thumb is to try to make your code blend in with the surrounding code.

Code
++++
 * 4 spaces for indentation (i.e. no tabs please)
 * 80 character line length
 * commas last
 * declare variables in the outermost scope that they are used
 * camelCase for variables in JavaScript and for classes/objects in Python
 * snake_case for variables in Python

Docstrings
++++++++++

Coding Rules
++++++++++++

To ensure consistency throughout the source code, keep these rules in mind as you are working:

* All features or bug fixes **must be tested** by one or more specs (unit-tests).
* We follow [Google's JavaScript Style Guide][js-style-guide].

-----

.. _community/guidelines/testing:

Testing
*******


-----
