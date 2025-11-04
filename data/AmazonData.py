from framework.data.DataSetup import DataSetup
from framework.data.Database import Database


class AmazonData(Database):
    def __init__(self, setup: DataSetup):
        super().__init__(setup)
        pass


    def data_for_labelling(self) -> list[tuple] | list[str]:
        import sqlite3
        try:
            conn = sqlite3.connect(self.setup.sqlite_attach_file)
            cursor = conn.cursor()
            columns = ["item_id","main_category"]
            cursor.execute(f"SELECT {",".join(columns)} FROM products ORDER BY item_id")
            rows = cursor.fetchall()
            conn.close()
            return rows, columns
            pass
        except Exception as e:
            print(self.setup.sqlite_attach_file)
            print(e)
            return [],[]

    def data_for_embedding(self) -> list[tuple] | list[str]:
        import sqlite3
        try:
            conn = sqlite3.connect(self.setup.sqlite_attach_file)
            cursor = conn.cursor()
            columns = ["item_id","title","features","description"]
            cursor.execute(f"SELECT {",".join(columns)} FROM products ORDER BY item_id")
            rows = cursor.fetchall()
            conn.close()
            return rows, columns
            pass
        except Exception as e:
            print(self.setup.sqlite_attach_file)
            print(e)
            return [],[]

    def data_for_finetune(self) -> list[tuple] | list[str]:
        pass

    def data_of_item(self, item_id) -> tuple | list[str]:
        import sqlite3
        try:
            conn = sqlite3.connect(self.setup.sqlite_attach_file)
            cursor = conn.cursor()
            columns = ["item_id","main_category","sub_categories","title","features","description","image"]
            cursor.execute(f"SELECT {",".join(columns)} FROM products WHERE item_id={item_id} LIMIT 1")
            rows = cursor.fetchall()
            conn.close()
            if len(rows) < 1:
                return (),[]
            return rows[0], columns
            pass
        except Exception as e:
            print(self.setup.sqlite_attach_file)
            print(e)
            return (),[]
        pass