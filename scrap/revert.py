from onedl.client import connect_to_project

client = connect_to_project('daniel-osman---streamlining-annotation-bootstrapping/pipeline-test')

past = client.datasets.load('pseudo-f0:1')
new_version_name = client.datasets.save('pseudo-f0', past, exist="versions")
client.datasets.push(new_version_name, push_policy="version")
